import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import threading
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE SORGENTI ---
# Modifica qui con i tuoi video o webcam (0, 1, 2)
SOURCES_CONFIG = {
    "Direttore": "Video\director.mp4",              # Webcam
    "Orchestra": "Video\musician.mp4",  # File video di test
    "Pubblico":  'Video\Audience.mp4'   # File video di test
}

# Lista emozioni per il prompt
VALID_EMOTIONS = "joy, anger, fear, disgust, surprise, sadness, boredom, neutral"

class MultiFrameData:
    def __init__(self):
        self.latest_frames = {k: None for k in SOURCES_CONFIG.keys()}
        self.latest_emotions = {k: "Inizializzazione..." for k in SOURCES_CONFIG.keys()}
        self.lock = threading.Lock()
        self.run_analysis = True

# --- WORKER MOONDREAM ---
def analysis_worker(data, model, tokenizer): 
    print("🧠 Thread Moondream avviato.")
    
    # Prompt specifico per Moondream
    # Moondream capisce bene il linguaggio naturale.
    question = f"Describe the general emotion of the people. Choose one word from: {VALID_EMOTIONS}."

    while data.run_analysis:
        frames_to_process = {}
        
        # 1. Copia sicura dei frame
        with data.lock:
            for key, frame in data.latest_frames.items():
                if frame is not None:
                    frames_to_process[key] = frame.copy()
        
        if frames_to_process:
            try:
                start = time.time()
                results = {}

                # 2. Ciclo di inferenza (Sequenziale ma veloce su Moondream)
                for key, frame_bgr in frames_to_process.items():
                    # Conversione immagine
                    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    
                    # Moondream step 1: Encoding immagine
                    # (Nota: encode_image gestisce automaticamente il device)
                    enc_image = model.encode_image(image)
                    
                    # Moondream step 2: Risposta alla domanda
                    # answer_question è un metodo helper specifico di Moondream
                    answer = model.answer_question(enc_image, question, tokenizer)
                    
                    # Pulizia base dell'output
                    results[key] = answer.strip()

                print(f"Cycle Time: {time.time() - start:.2f}s | Results: {results}")

                # 3. Aggiornamento dati condivisi
                with data.lock:
                    for key, val in results.items():
                        data.latest_emotions[key] = val
            
            except Exception as e:
                print(f"‼️ Errore Moondream: {e}")
        
        # Sleep leggero per non intasare la CPU/GPU
        time.sleep(0.1) 
    print("🛑 Thread analisi terminato.")

def main_moondream_monitor():
    print("⏳ Caricamento Moondream2 (può richiedere qualche minuto al primo avvio)...")
    
    # Usiamo una revisione specifica per stabilità
    model_id = "vikhyatk/moondream2"    

    try:
        # Moondream gira bene anche su CPU se non hai CUDA, ma CUDA è meglio.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️  Dispositivo rilevato: {device.upper()}")

        # Caricamento Modello
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,            
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": device} # Forza il device
        )
        
        # Caricamento Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("✅ Moondream caricato!")

    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return

    # --- SETUP VIDEO (Uguale a prima) ---
    captures = {}
    for name, source in SOURCES_CONFIG.items():
        # Se è un file video e non esiste, OpenCV non darà errore subito ma dopo
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"✅ Camera/Video '{name}' ok.")
            captures[name] = cap
        else:
            print(f"⚠️ Sorgente '{name}' non trovata (controlla webcam index o path file).")
    
    if not captures:
        print("❌ Nessuna sorgente valida. Esco.")
        return

    data = MultiFrameData()
    
    # Avvio Thread Analisi
    analyzer_thread = threading.Thread(target=analysis_worker, args=(data, model, tokenizer))
    analyzer_thread.start()

    # --- GUI MATPLOTLIB ---
    plt.ion()
    # Griglia dinamica in base alle sorgenti trovate
    n_sources = len(captures)
    fig, axes = plt.subplots(1, n_sources, figsize=(5 * n_sources, 5))
    
    # Normalizza axes in una lista anche se è uno solo
    if n_sources == 1: axes = [axes]
    
    # Associa asse a nome sorgente
    axes_map = {name: ax for name, ax in zip(captures.keys(), axes)}

    print("\n🚀 Monitoraggio avviato. Premi CTRL+C nella console o chiudi la finestra per uscire.")

    try:
        while True:
            current_frames = {}
            
            # Lettura frame dai video
            for name, cap in captures.items():
                ret, frame = cap.read()
                if ret:
                    # Loop video per test infiniti (se è un file)
                    if isinstance(SOURCES_CONFIG[name], str): 
                        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                    current_frames[name] = frame
                else:
                    current_frames[name] = None

            # Aggiorna frame per l'analizzatore
            with data.lock:
                for name, frame in current_frames.items():
                    if frame is not None:
                        data.latest_frames[name] = frame
                emotions = data.latest_emotions.copy()

            # Disegno
            for name, ax in axes_map.items():
                ax.clear()
                frame = current_frames.get(name)
                
                if frame is not None:
                    # BGR -> RGB
                    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    emo = emotions.get(name, "...")
                    
                    # Titolo colorato
                    ax.set_title(f"{name}", fontsize=12, fontweight='bold')
                    ax.text(10, 30, emo, fontsize=14, color='white', 
                            bbox=dict(facecolor='red', alpha=0.7))
                else:
                    ax.text(0.5, 0.5, "NO VIDEO", ha='center')
                
                ax.axis('off')

            plt.pause(0.001)
            
            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        print("Interruzione manuale...")
    except Exception as e:
        print(f"Errore runtime: {e}")
    finally:
        data.run_analysis = False
        analyzer_thread.join()
        for cap in captures.values():
            cap.release()
        plt.ioff()
        print("👋 Chiusura completata.")

if __name__ == "__main__":
    main_moondream_monitor()