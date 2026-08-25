import cv2
import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import time
import threading
import matplotlib.pyplot as plt

# Lista emozioni (usata nel prompt per guidare il modello)
VALID_EMOTIONS = ['joy', 'anger', 'fear', 'disgust', 'surprise', 'sadness', 'boredom', 'neutral']

# --- CLASSE CONDIVISA PER SCAMBIARE I DATI TRA I "LAVORATORI" ---
class FrameData:
    def __init__(self):
        self.latest_frame = None
        self.latest_emotion = "Inizializzazione..."
        self.lock = threading.Lock() # Un "semaforo" per la sicurezza
        self.run_analysis = True

# --- FUNZIONE DI ANALISI (IL "PENSATORE" CHE GIRA IN BACKGROUND) ---
def analysis_worker(data, model, processor): 
    print("🧠 Thread di analisi avviato.")
    while data.run_analysis:
        frame_to_analyze = None
        
        # Prende l'ultimo frame disponibile in modo sicuro
        with data.lock:
            if data.latest_frame is not None:
                frame_to_analyze = data.latest_frame.copy()
        
        if frame_to_analyze is not None:
            try:
                # Conversione BGR (OpenCV) -> RGB (PIL)
                image = Image.fromarray(cv2.cvtColor(frame_to_analyze, cv2.COLOR_BGR2RGB))
                
                # Prompt ottimizzato per PaliGemma
                # PaliGemma funziona meglio con prompt diretti.
                prompt_text = (
                    f"<image> What is the general emotion in this picture? "
                    f"Answer the question using a single word for each emotion you can find. "
                    f"Follow this example: '[ em1, em2, em3 ]'.\n"
                    f"You MUST choose only from: {VALID_EMOTIONS}."
                )
                
                start = time.time()
                
                # 1. Preparazione degli input tramite il Processor
                model_inputs = processor(text=prompt_text, images=image, return_tensors="pt")
                
                # Sposta gli input sulla GPU e nel formato corretto (bfloat16)
                model_inputs = model_inputs.to(dtype=model.dtype, device=model.device)
                
                # Calcola la lunghezza dell'input per tagliare via il prompt dalla risposta
                input_len = model_inputs["input_ids"].shape[-1]

                # 2. Generazione della risposta
                with torch.inference_mode():
                    generation = model.generate(
                        **model_inputs, 
                        max_new_tokens=20, # Teniamo basso per velocità (ci serve solo una parola)
                        do_sample=False    # Deterministico è solitamente meglio per classificazione
                    )
                    
                # 3. Decodifica (rimuovendo la parte del prompt input)
                generation = generation[0][input_len:]
                decoded_output = processor.decode(generation, skip_special_tokens=True)
                
                # Pulizia stringa (rimuove spazi extra o newlines)
                model_output = decoded_output.strip()

                print(f"Output Modello: {model_output} | Tempo: {time.time() - start:.2f}s")

                # Aggiorna il risultato in modo sicuro
                with data.lock:
                    data.latest_emotion = model_output
            
            except Exception as e:
                print(f"‼️ Errore nel thread di analisi: {e}")
        
        # Pausa per non saturare la GPU
        time.sleep(0.5) 
    print("🛑 Thread di analisi terminato.")

# --- FUNZIONE PRINCIPALE (IL "VISORE" CHE GESTISCE IL VIDEO) ---
def main_realtime_matplotlib():

    print("🔑 Eseguo il login a Hugging Face...")
    try:
        hf_token = "YOUR_HF_TOKEN" # Removed for security
        huggingface_hub.login(token=hf_token)
        print("✅ Login a Hugging Face Hub effettuato con successo!")
    except Exception as e:
        print(f"‼️ Errore durante il login a Hugging Face: {e}")
        print("Il programma continuerà, ma potrebbe fallire se il modello non è già in cache.")

    print("⏳ Caricamento del modello PaliGemma 2...")
    
    model_id = "google/paligemma2-3b-mix-448"
    
    # Caricamento Modello
    # Usiamo device_map="auto" per gestire automaticamente la GPU
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        #revision="fp16" # Opzionale, a volte aiuta a scaricare i pesi giusti
    ).eval()

    # Caricamento Processor
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    print("✅ Modello caricato!")

    data = FrameData()
    # Su Windows/Linux indice 0 per webcam
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("❌ Impossibile aprire la camera.")
        return

    # Crea e avvia il thread "Pensatore"
    # Passiamo 'processor' invece di 'tokenizer'
    analyzer_thread = threading.Thread(target=analysis_worker, args=(data, model, processor))
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
            # Conversione colore per visualizzazione corretta
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)
            
            # Overlay del testo
            ax.text(30, 70, f"PaliGemma vede: {emotion_to_display}", 
                    fontsize=20, color='lime', backgroundcolor='black')
            ax.axis('off')
            
            # Pausa per aggiornamento GUI
            plt.pause(0.001)

            # Controllo chiusura finestra
            if not plt.fignum_exists(fig.number):
                data.run_analysis = False
                break
    except KeyboardInterrupt:
        print("Interruzione manuale...")
    finally:
        # Pulizia finale
        print("🔌 Chiusura in corso...")
        data.run_analysis = False
        analyzer_thread.join()
        cap.release()
        plt.ioff()
        print("👋 Applicazione terminata.")

if __name__ == "__main__":
    main_realtime_matplotlib()