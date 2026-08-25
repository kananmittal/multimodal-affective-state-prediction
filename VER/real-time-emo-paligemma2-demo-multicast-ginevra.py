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
import cv2

# ================= CONFIGURAZIONE =================
# 1. PERCORSO MODELLO
MODEL_PATH = "../paligemma_offline"

# 2. RISOLUZIONE (⚠️ DEVE ESSERE IDENTICA A OBS PER OGNI FLUSSO!)
WIDTH = 1280
HEIGHT = 720

# 3. CONFIGURAZIONE SORGENTI (3 Porte diverse)
SOURCES_CONFIG = [
    {"name": "Direttore", "port": 9000, "color": "lime"},
    {"name": "Orchestra", "port": 9001, "color": "cyan"},
    {"name": "Pubblico",  "port": 9002, "color": "magenta"}
]

VALID_EMOTIONS = ['joy', 'anger', 'fear', 'disgust', 'surprise', 'sadness', 'boredom', 'neutral']
# ==================================================

class MultiFrameData:
    def __init__(self):
        # Creiamo un dizionario per contenere i frame di ogni sorgente
        self.frames = {src["name"]: None for src in SOURCES_CONFIG}
        self.emotions = {src["name"]: "In attesa..." for src in SOURCES_CONFIG}
        self.lock = threading.Lock()
        self.run_analysis = True

# --- WORKER VIDEO GENERICO (Ne lanceremo 3) ---
def video_worker_pipe(source_conf, shared_data):
    name = source_conf["name"]
    port = source_conf["port"]
    
    # URL SRT specifico per questa porta
    srt_url = f"srt://0.0.0.0:{port}?mode=listener&latency=6000000&recv_buffer_size=10000000&maxbw=5000000"
    
    print(f"👀 [{name}] Avvio Pipe su porta {port}...")

    """
    cmd = [
        'ffmpeg',
        '-re', '-loglevel', 'error',
        '-i', srt_url,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-'
    ]"""

    cmd = [
        'ffmpeg',
        '-re',                         # Lettura real-time
        '-thread_queue_size', '4096',  # <--- FONDAMENTALE: Aumenta la coda di ingresso
        '-loglevel', 'error',
        '-fflags', '+genpts+discardcorrupt', # <--- Pulisce il segnale da errori
        #'-fflags', 'nobuffer+discardcorrupt', # <--- NO BUFFER: Riduce latenza interna
        #'-flags', 'low_delay',
        '-i', srt_url,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-'
    ]

    try:
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    except Exception as e:
        print(f"❌ [{name}] Errore FFmpeg: {e}")
        return

    frame_size = WIDTH * HEIGHT * 3
    
    while shared_data.run_analysis:
        raw_image = pipe.stdout.read(frame_size)
        
        if len(raw_image) != frame_size:
            time.sleep(0.01) # Buffering o attesa connessione
            continue
            
        try:
            image = np.frombuffer(raw_image, dtype='uint8')
            frame = image.reshape((HEIGHT, WIDTH, 3))
            
            with shared_data.lock:
                shared_data.frames[name] = frame
        except Exception as e:
            pass

    pipe.terminate()
    print(f"⏹️ [{name}] Pipe chiuso.")

# --- WORKER ANALISI (Cervello Unico Round-Robin) ---
def analysis_worker(data, model, processor): 
    print("🧠 [VLM] Thread analisi Multi-Stream partito.")
    
    while data.run_analysis:
        # 1. Cattura snapshot dei frame attuali
        current_frames = {}
        with data.lock:
            # Facciamo una copia per non bloccare i video worker
            for name, frame in data.frames.items():
                if frame is not None:
                    current_frames[name] = frame.copy()
        
        # 2. Se non c'è nessun video, attendi
        if not current_frames:
            time.sleep(0.1)
            continue

        # 3. Analizza ogni frame disponibile in sequenza
        for name, frame in current_frames.items():
            try:
                # Resize
                frame_resized = cv2.resize(frame, (448, 448))
                image = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                
                prompt_text = f"<image>Detect the main emotion. Choose ONE from: {VALID_EMOTIONS}."
                
                model_inputs = processor(text=prompt_text, images=image, return_tensors="pt")
                model_inputs = model_inputs.to(dtype=model.dtype, device=model.device)
                input_len = model_inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = model.generate(**model_inputs, max_new_tokens=10, do_sample=False)
                    
                output = processor.decode(generation[0][input_len:], skip_special_tokens=True).strip()

                with data.lock:
                    data.emotions[name] = output
                
                print(f"--> {name}: {output}")
                
            except Exception as e:
                print(f"‼️ Err {name}: {e}")

        # Piccola pausa tra un ciclo completo di 3 camere e l'altro
        time.sleep(0.05)

# --- MAIN ---
def main():
    print("--- SERVER AI MULTI-STREAM (3 Porte SRT) ---")
    
    # Caricamento Modello
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

    shared_data = MultiFrameData()
    
    # Avvio 3 Thread Video (uno per porta)
    video_threads = []
    for source_conf in SOURCES_CONFIG:
        t = threading.Thread(target=video_worker_pipe, args=(source_conf, shared_data))
        t.daemon = True
        t.start()
        video_threads.append(t)
    
    time.sleep(1) # Attesa tecnica avvio FFmpeg

    # Avvio Thread Analisi
    t_vlm = threading.Thread(target=analysis_worker, args=(shared_data, model, processor))
    t_vlm.daemon = True
    t_vlm.start()

    # --- GUI MATPLOTLIB (3 Monitor) ---
    print("🚀 Apro Monitor di Controllo...")
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    # Setup grafica iniziale
    displays = {}
    texts = {}
    
    for idx, conf in enumerate(SOURCES_CONFIG):
        ax = axes[idx]
        name = conf["name"]
        color = conf["color"]
        
        dummy = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        displays[name] = ax.imshow(dummy)
        texts[name] = ax.text(50, 100, f"ATTESA {conf['port']}...", 
                              color=color, fontsize=14, backgroundcolor='black', weight='bold')
        ax.set_title(f"{name} (Porta {conf['port']})", color='white', backgroundcolor='black')
        ax.axis('off')

    try:
        while True:
            # Lettura sicura dati
            with shared_data.lock:
                frames_snapshot = shared_data.frames.copy()
                emotions_snapshot = shared_data.emotions.copy()
            
            # Aggiornamento GUI
            for name, frame in frames_snapshot.items():
                if frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    displays[name].set_data(frame_rgb)
                    
                    emo = emotions_snapshot[name]
                    texts[name].set_text(f"{emo.upper()}")
            
            fig.canvas.flush_events()
            
            if not plt.fignum_exists(fig.number):
                break
            
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stop manuale.")
    finally:
        shared_data.run_analysis = False
        t_vlm.join()
        for t in video_threads:
            t.join()
        plt.close()

if __name__ == "__main__":
    main()