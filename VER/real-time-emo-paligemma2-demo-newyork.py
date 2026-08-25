import sys
import time
import threading
import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

# ================= CONFIGURAZIONE =================
# 1. PERCORSO MODELLO
#MODEL_PATH = os.path.join(os.getcwd(), "paligemma_offline")
MODEL_PATH = "../paligemma_offline"




# 2. CONFIGURAZIONE STREAMING (SRT)
# 0.0.0.0 = Ascolta su tutte le reti (inclusa ZeroTier)
# mode=listener = Noi siamo il server, aspettiamo che OBS ci chiami
# latency=2000000 = 2 secondi di buffer per evitare scatti via internet

#STREAM_URL = "udp://0.0.0.0:9000"
STREAM_URL = "srt://0.0.0.0:12345?mode=listener&latency=500000"

VALID_EMOTIONS = ['joy', 'anger', 'fear', 'disgust', 'surprise', 'sadness', 'boredom', 'neutral']
# ==================================================

class FrameData:
    def __init__(self):
        self.latest_frame = None
        self.latest_emotion = "In attesa di video..."
        self.lock = threading.Lock()
        self.run_analysis = True

# --- THREAD 1: ANALISI VLM (Il Cervello) ---
def analysis_worker(data, model, processor): 
    print("🧠 [VLM] Thread analisi partito.")
    while data.run_analysis:
        frame_to_analyze = None
        
        with data.lock:
            if data.latest_frame is not None:
                frame_to_analyze = data.latest_frame.copy()
        
        if frame_to_analyze is not None:
            try:
                # Resize per PaliGemma
                frame_resized = cv2.resize(frame_to_analyze, (448, 448))
                image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                
                prompt_text = (
                    f"<image>Detect the main emotion. "
                    f"Choose ONE from: {VALID_EMOTIONS}."
                )
                
                model_inputs = processor(text=prompt_text, images=image, return_tensors="pt")
                model_inputs = model_inputs.to(dtype=model.dtype, device=model.device)
                input_len = model_inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = model.generate(
                        **model_inputs, max_new_tokens=10, do_sample=False
                    )
                    
                output = processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()

                with data.lock:
                    data.latest_emotion = output
                
                # Piccola pausa per non fondere la GPU se l'inferenza è troppo veloce
                time.sleep(0.05)
                    
            except Exception as e:
                print(f"‼️ VLM Error: {e}")
                time.sleep(0.5)
        else:
            time.sleep(0.1)

# --- THREAD 2: RICEZIONE VIDEO VIA SRT (L'Occhio) ---
def video_worker(shared_data):
    print(f"👀 [VIDEO] Avvio Server SRT...")
    print(f"    --> Indirizzo: {STREAM_URL}")
    print("    --> ORA vai sul MAC (OBS) e premi 'Avvia Registrazione'!")
    
    # Usiamo CAP_FFMPEG perché è un flusso di rete, non una webcam USB
    #cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_FFMPEG)
    cap = cv2.VideoCapture(STREAM_URL, cv2.CAP_GSTREAMER)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    if not cap.isOpened():
        print("❌ ERRORE CRITICO: Impossibile aprire la porta 9000.")        
        shared_data.run_analysis = False
        return

    print("✅ CONNESSIONE STABILITA! Sto ricevendo le immagini.")
    
    while shared_data.run_analysis:
        ret, frame = cap.read()
        if ret:
            with shared_data.lock:
                shared_data.latest_frame = frame
        else:
            # Se la connessione salta o c'è buffering, aspettiamo senza crashare
            print(".", end='', flush=True)
            time.sleep(0.05)
            
            # Se si disconnette del tutto, proviamo a riaprire (opzionale)
            if not cap.isOpened():
                print("\n🔄 Tentativo riconnessione...")
                cap.open(STREAM_URL, cv2.CAP_FFMPEG)

    cap.release()
    print("\n⏹️ Stream chiuso.")

# --- MAIN ---
def main():
    print("--- SERVER AI: RICEZIONE DA NEW YORK (Simulato) ---")

    # 1. CARICAMENTO MODELLO
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRORE: Cartella {MODEL_PATH} non trovata.")
        return

    try:
        print(f"📂 Carico modello PaliGemma...")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            dtype=torch.bfloat16, 
            device_map="auto",
            local_files_only=True 
        ).eval()
        processor = PaliGemmaProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        print("✅ Modello caricato.")
    except Exception as e:
        print(f"❌ Errore VLM: {e}")
        return

    shared_data = FrameData()
    
    # 2. AVVIO RICEZIONE VIDEO
    # Parte prima il thread video così apre la porta e si mette in ascolto
    t_video = threading.Thread(target=video_worker, args=(shared_data,))
    t_video.daemon = True
    t_video.start()
    
    # Aspettiamo un attimo per sicurezza
    time.sleep(1)

    # 3. AVVIO ANALISI INTELLIGENTE
    t_vlm = threading.Thread(target=analysis_worker, args=(shared_data, model, processor))
    t_vlm.daemon = True
    t_vlm.start()

    # 4. VISUALIZZAZIONE
    print("🚀 Apro finestra...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    im_display = ax.imshow(dummy_img)
    text_display = ax.text(10, 30, "In attesa connessione...", color='lime', fontsize=14, backgroundcolor='black')
    ax.axis('off')

    try:
        while True:
            frame = None
            emotion = ""
            with shared_data.lock:
                if shared_data.latest_frame is not None:
                    frame = shared_data.latest_frame.copy()
                    emotion = shared_data.latest_emotion

            if frame is not None:
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_display.set_data(frame_rgb)
                text_display.set_text(f"AI Vede: {emotion}")
                
                # Titolo finestra
                fig.canvas.manager.set_window_title(f"Live Analysis: {emotion}")
                fig.canvas.flush_events()
            
            if not plt.fignum_exists(fig.number):
                break
            
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Stop manuale.")
    finally:
        shared_data.run_analysis = False
        t_vlm.join()
        t_video.join()
        plt.close()
        print("Chiusura.")

if __name__ == "__main__":
    main()
