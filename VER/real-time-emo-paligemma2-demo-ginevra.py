import sys
import time
import threading
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import torch
import cv2  # Importiamo cv2 globalmente per la GUI

# ================= CONFIGURAZIONE =================
# 1. PERCORSO MODELLO
MODEL_PATH = "../paligemma_offline"

# 2. RISOLUZIONE (⚠️ DEVE ESSERE IDENTICA A OBS!)
WIDTH = 1280
HEIGHT = 720

# 3. URL SRT (Porta 12345)
SRT_URL = "srt://0.0.0.0:9000?mode=listener&latency=500000"

VALID_EMOTIONS = ['joy', 'anger', 'fear', 'disgust', 'surprise', 'sadness', 'boredom', 'neutral']
# ==================================================

class FrameData:
    def __init__(self):
        self.latest_frame = None
        self.latest_emotion = "In attesa..."
        self.lock = threading.Lock()
        self.run_analysis = True

# --- THREAD VIDEO: METODO "PIPE" (BLINDATO) ---
def video_worker(shared_data):
    print(f"👀 [VIDEO] Avvio Pipe FFmpeg su porta 12345...")
    print(f"    --> Aspetto risoluzione: {WIDTH}x{HEIGHT}")
    
    cmd = [
        'ffmpeg',
        '-re',                        # Real-time
        '-loglevel', 'error',         # Silenzioso
        '-i', SRT_URL,                # Input (SRT su porta 12345)
        '-f', 'image2pipe',           # Output su Tubo
        '-pix_fmt', 'bgr24',          # Formato Pixel per OpenCV
        '-vcodec', 'rawvideo',
        '-'                           # Output verso Python
    ]

    try:
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**5)
    except Exception as e:
        print(f"❌ Errore critico avvio FFmpeg: {e}")
        return

    print("✅ SERVER PRONTO: Ora premi 'Avvia Registrazione' su OBS!")
    
    frame_size = WIDTH * HEIGHT * 3
    
    while shared_data.run_analysis:
        raw_image = pipe.stdout.read(frame_size)
        
        if len(raw_image) != frame_size:
            time.sleep(0.01)
            continue
            
        try:
            image = np.frombuffer(raw_image, dtype='uint8')
            frame = image.reshape((HEIGHT, WIDTH, 3))
            
            with shared_data.lock:
                shared_data.latest_frame = frame
        except Exception as e:
            pass

    pipe.terminate()
    print("⏹️ Pipe chiuso.")

# --- THREAD ANALISI VLM ---
def analysis_worker(data, model, processor): 
    print("🧠 [VLM] Thread analisi partito.")
    while data.run_analysis:
        frame_to_analyze = None
        with data.lock:
            if data.latest_frame is not None:
                frame_to_analyze = data.latest_frame.copy()
        
        if frame_to_analyze is not None:
            try:
                frame_resized = cv2.resize(frame_to_analyze, (448, 448))
                image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                
                prompt_text = f"<image>Detect the main emotion. Choose ONE from: {VALID_EMOTIONS}."
                
                model_inputs = processor(text=prompt_text, images=image, return_tensors="pt")
                model_inputs = model_inputs.to(dtype=model.dtype, device=model.device)
                input_len = model_inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = model.generate(**model_inputs, max_new_tokens=10, do_sample=False)
                    
                output = processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()

                with data.lock:
                    data.latest_emotion = output
                
                print(f"--> VEDO: {output}")
                time.sleep(0.05)
            except Exception as e:
                print(f"‼️ VLM Error: {e}")
        else:
            time.sleep(0.1)

# --- MAIN (ORA CON GUI!) ---
def main():
    print("--- SERVER AI: METODO PIPE CON VIDEO ---")
    
    # Caricamento modello
    try:
        print(f"📂 Carico PaliGemma...")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="auto", local_files_only=True 
        ).eval()
        processor = PaliGemmaProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        print("✅ Modello caricato.")
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return

    shared_data = FrameData()
    
    # Avvio threads
    t_video = threading.Thread(target=video_worker, args=(shared_data,))
    t_video.daemon = True
    t_video.start()
    
    time.sleep(1)

    t_vlm = threading.Thread(target=analysis_worker, args=(shared_data, model, processor))
    t_vlm.daemon = True
    t_vlm.start()

    # --- CONFIGURAZIONE INTERFACCIA GRAFICA ---
    print("🚀 Apro finestra Video...")
    plt.ion()  # Modalità interattiva
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Immagine nera di partenza
    dummy_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    im_display = ax.imshow(dummy_img)
    
    # Testo sovraimpresso
    text_display = ax.text(50, 100, "ATTESA SEGNALE...", 
                           color='lime', fontsize=20, backgroundcolor='black', weight='bold')
    
    ax.axis('off') # Nasconde gli assi numerici
    
    try:
        while True:
            # Preleviamo i dati dal thread in modo sicuro
            frame = None
            emotion = ""
            with shared_data.lock:
                if shared_data.latest_frame is not None:
                    frame = shared_data.latest_frame.copy()
                    emotion = shared_data.latest_emotion

            # Se abbiamo un frame, aggiorniamo la grafica
            if frame is not None:
                # Matplotlib vuole RGB, OpenCV ci dà BGR -> Convertiamo
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Aggiorna l'immagine
                im_display.set_data(frame_rgb)
                
                # Aggiorna il testo
                text_display.set_text(f"EMOZIONE: {emotion.upper()}")
                
                # Aggiorna titolo finestra
                fig.canvas.manager.set_window_title(f"AI Live Monitor - {emotion}")
                
                # Disegna effettivamente
                fig.canvas.flush_events()
            
            # Controllo se la finestra è stata chiusa dall'utente
            if not plt.fignum_exists(fig.number):
                print("Finestra chiusa dall'utente.")
                break
            
            # Pausa tecnica per dare tempo alla grafica di aggiornarsi
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stop manuale.")
    finally:
        shared_data.run_analysis = False
        t_vlm.join()
        t_video.join()
        plt.close()

if __name__ == "__main__":
    main()