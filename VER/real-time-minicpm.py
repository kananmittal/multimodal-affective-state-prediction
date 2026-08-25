import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer 
import time
import threading
import matplotlib.pyplot as plt


VALID_EMOTIONS = ['joy', 'anger', 'fear', 'disgust', 'surprise', 'sadness', 'boredom', 'neutral']
# --- CLASSE CONDIVISA PER SCAMBIARE I DATI TRA I "LAVORATORI" ---
class FrameData:
    def __init__(self):
        self.latest_frame = None
        self.latest_emotion = "Inizializzazione..."
        self.lock = threading.Lock() # Un "semaforo" per la sicurezza
        self.run_analysis = True

# --- FUNZIONE DI ANALISI (IL "PENSATORE" CHE GIRA IN BACKGROUND) ---
# --- MODIFICATO ---
# Rinominato 'processor' in 'tokenizer' per coerenza
def analysis_worker(data, model, tokenizer): 
    print("🧠 Thread di analisi avviato.")
    while data.run_analysis:
        frame_to_analyze = None
        
        # Prende l'ultimo frame disponibile in modo sicuro
        with data.lock:
            if data.latest_frame is not None:
                frame_to_analyze = data.latest_frame.copy()
        
        if frame_to_analyze is not None:
            try:
                # --- BLOCCO DI ANALISI MODIFICATO CON IL METODO .chat() ---
                image = Image.fromarray(cv2.cvtColor(frame_to_analyze, cv2.COLOR_BGR2RGB))
                
                prompt_text = (
                    f"What is the general emotion in this picture? "
                    f"Answer the question using a single word for each emotion you can find. "
                    f"Follow this example: '[ em1, em2, em3 ]'.\n"
                    f"You MUST choose only from: {VALID_EMOTIONS}."
                )
                
                # Come nel tuo esempio, creiamo 'msgs' con immagine e testo
                # NOTA: il prompt di MiniCPM-V non vuole <image> quando si usa .chat()
                msgs = [{'role': 'user', 'content': [image, prompt_text]}]

                start = time.time()
                
                # Usiamo il metodo .chat() del modello, che fa tutto lui!
                model_output = model.chat(
                    image=image, # L'immagine va passata anche qui
                    msgs=msgs,
                    tokenizer=tokenizer
                )
                # --- FINE BLOCCO MODIFICATO ---
                
                print(f"Output Modello: {model_output}")
                print(f"Tempo inferenza: {time.time() - start}")

                # Aggiorna il risultato in modo sicuro
                with data.lock:
                    data.latest_emotion = model_output
                print(f"🔎 Emozione Rilevata: {model_output}")
            except Exception as e:
                print(f"‼️ Errore nel thread di analisi: {e}")
        
        # Pausa per non fare analisi troppo ravvicinate
        #time.sleep(.5) 
    print("🛑 Thread di analisi terminato.")

# --- FUNZIONE PRINCIPALE (IL "VISORE" CHE GESTISCE IL VIDEO) ---
def main_realtime_matplotlib():
    print("⏳ Caricamento del modello MiniCPM-V (metodo .chat)...")
    
    # --- MODIFICATO ---
    # Uso i parametri del tuo snippet
    model_id = "openbmb/MiniCPM-V-4"
    model = AutoModel.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, # Ottimo per GPU recenti
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        attn_implementation='sdpa'  # Ottimizzazione
    ).to("cuda").eval() # Aggiunto .eval() per modalità inferenza
    
    # Usiamo AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True
    )
    print("✅ Modello caricato!")

    data = FrameData()
    # Su Windows, usa l'indice 0 per la webcam predefinita
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2) 
    if not cap.isOpened():
        print("❌ Impossibile aprire la camera.")
        return

    # Crea e avvia il thread "Pensatore"
    # --- MODIFICATO ---
    # Passiamo 'tokenizer'
    analyzer_thread = threading.Thread(target=analysis_worker, args=(data, model, tokenizer))
    analyzer_thread.start()

    # Prepara la finestra di Matplotlib
    plt.ion()
    fig, ax = plt.subplots(figsize=(16, 9))
    print("\n🚀 Applicazione avviata! Chiudi la finestra di Matplotlib per uscire.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Passa l'ultimo frame al "Pensatore" e prende l'ultima emozione calcolata
            with data.lock:
                data.latest_frame = frame
                emotion_to_display = data.latest_emotion

            # Disegna la scena sulla finestra di Matplotlib
            ax.clear()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)
            ax.text(30, 70, f"Emozione: {emotion_to_display}", 
                    fontsize=20, color='lime', backgroundcolor='black')
            ax.axis('off')
            
            # Pausa infinitesimale per permettere alla finestra di aggiornarsi
            plt.pause(0.001)

            # Se l'utente ha chiuso la finestra, fermiamo tutto
            if not plt.fignum_exists(fig.number):
                data.run_analysis = False
                break
    finally:
        # Pulizia finale, assicurandosi che tutto si chiuda correttamente
        print("🔌 Chiusura in corso...")
        data.run_analysis = False
        analyzer_thread.join() # Aspetta che il "Pensatore" finisca il suo ultimo giro
        cap.release()
        plt.ioff()
        print("👋 Applicazione terminata.")

if __name__ == "__main__":
    main_realtime_matplotlib()