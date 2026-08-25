import cv2
import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import time
import threading
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE SORGENTI ---
# Definisci qui le tue sorgenti: possono essere indici (0, 1) per webcam o percorsi file "video.mp4"
SOURCES_CONFIG = {
    "Direttore": 0,          # Esempio: Webcam 1
    "Orchestra": "video_orch.mp4", # Esempio: File video (o metti 1 per seconda webcam)
    "Pubblico":  "video_pub.mp4"   # Esempio: File video (o metti 2 per terza webcam)
}

VALID_EMOTIONS = ['joy', 'anger', 'fear', 'disgust', 'surprise', 'sadness', 'boredom', 'neutral']

class MultiFrameData:
    def __init__(self):
        # Dizionario per salvare l'ultimo frame di ogni sorgente
        self.latest_frames = {k: None for k in SOURCES_CONFIG.keys()}
        # Dizionario per salvare l'ultima emozione di ogni sorgente
        self.latest_emotions = {k: "In attesa..." for k in SOURCES_CONFIG.keys()}
        self.lock = threading.Lock()
        self.run_analysis = True

def analysis_worker(data, model, processor): 
    print("🧠 Thread di analisi Multi-Stream avviato.")
    
    # Prompt base (identico per tutti, ma potresti personalizzarlo per sorgente se volessi)
    base_prompt = (
        f"<image> What is the general emotion? Answer with a single word from: {VALID_EMOTIONS}."
    )

    while data.run_analysis:
        frames_batch = []
        keys_batch = []
        
        # 1. Raccolta sicura dei frame disponibili
        with data.lock:
            for key, frame in data.latest_frames.items():
                if frame is not None:
                    frames_batch.append(frame.copy())
                    keys_batch.append(key)
        
        # Se abbiamo almeno un frame da analizzare
        if frames_batch:
            try:
                # Conversione di tutti i frame in PIL Images
                pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_batch]
                
                # Creiamo una lista di prompt lunga quanto le immagini (Batching)
                prompts = [base_prompt] * len(pil_images)

                start = time.time()
                
                # --- INFERENZA IN BATCH (La magia avviene qui) ---
                # Il modello processa N immagini contemporaneamente
                model_inputs = processor(text=prompts, images=pil_images, return_tensors="pt", padding=True)
                model_inputs = model_inputs.to(dtype=model.dtype, device=model.device)
                
                input_len = model_inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = model.generate(
                        **model_inputs, 
                        max_new_tokens=10, 
                        do_sample=False
                    )
                
                # Decodifica dei risultati multipli
                results = []
                for i in range(len(generation)):
                    decoded = processor.decode(generation[i][input_len:], skip_special_tokens=True).strip()
                    results.append(decoded)

                print(f"Batch Processed ({len(results)} sources) | Time: {time.time() - start:.2f}s | Results: {results}")

                # Aggiornamento risultati
                with data.lock:
                    for idx, key in enumerate(keys_batch):
                        data.latest_emotions[key] = results[idx]
            
            except Exception as e:
                print(f"‼️ Errore analisi batch: {e}")
        
        # Pausa per evitare surriscaldamento (aggiusta in base alla potenza GPU)
        time.sleep(0.2) 
    print("🛑 Thread di analisi terminato.")

def main_multistream():
    # --- SETUP MODELLO (Identico a prima) ---
    print("⏳ Caricamento PaliGemma 2...")
    # Sostituisci col tuo token se necessario o usa login cli
    # huggingface_hub.login(token="...") 
    
    model_id = "google/paligemma2-3b-mix-448"
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    print("✅ Modello caricato!")

    # --- SETUP VIDEO ---
    captures = {}
    for name, source in SOURCES_CONFIG.items():
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print(f"✅ Sorgente '{name}' aperta.")
            captures[name] = cap
        else:
            print(f"❌ Impossibile aprire sorgente '{name}'.")
    
    if not captures:
        print("Nessuna sorgente video valida.")
        return

    data = MultiFrameData()
    analyzer_thread = threading.Thread(target=analysis_worker, args=(data, model, processor))
    analyzer_thread.start()

    # --- SETUP GRAFICA (3 Subplots) ---
    plt.ion()
    # Crea una griglia 1 riga x 3 colonne
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    if len(SOURCES_CONFIG) == 1: axes = [axes] # Gestione caso singola sorgente per debug
    
    # Mappiamo i nomi degli assi per facilità
    axes_map = {name: ax for name, ax in zip(SOURCES_CONFIG.keys(), axes)}

    print("\n🚀 Multi-Stream avviato!")

    try:
        while True:
            # Lettura frame da tutte le sorgenti
            current_frames = {}
            for name, cap in captures.items():
                ret, frame = cap.read()
                if ret:
                    # Gestione loop video se usi file mp4 (opzionale)
                    # if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    current_frames[name] = frame
                else:
                    # Se una camera si stacca o video finisce
                    current_frames[name] = None

            # Aggiorna i dati per il thread di analisi
            with data.lock:
                for name, frame in current_frames.items():
                    if frame is not None:
                        data.latest_frames[name] = frame
                
                # Prende le emozioni attuali per visualizzarle
                emotions_snapshot = data.latest_emotions.copy()

            # Disegno grafico
            for name, ax in axes_map.items():
                ax.clear()
                frame = current_frames.get(name)
                
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ax.imshow(frame_rgb)
                    emo = emotions_snapshot.get(name, "...")
                    
                    # Colore diverso per ogni ruolo
                    color = 'lime'
                    if name == 'Orchestra': color = 'cyan'
                    if name == 'Pubblico': color = 'orange'

                    ax.set_title(f"{name}: {emo}", fontsize=15, color='white', backgroundcolor='black')
                else:
                    ax.text(0.5, 0.5, "NO SIGNAL", ha='center', va='center')
                
                ax.axis('off')

            plt.pause(0.001)

            if not plt.fignum_exists(fig.number):
                break

    except KeyboardInterrupt:
        print("Interruzione...")
    finally:
        data.run_analysis = False
        analyzer_thread.join()
        for cap in captures.values():
            cap.release()
        plt.ioff()
        print("👋 Fine.")

if __name__ == "__main__":
    main_multistream()