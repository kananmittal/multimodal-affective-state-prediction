import sys
import time
import threading
import numpy as np
import os
import subprocess
import datetime
import random
from flask import Flask, Response
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

# ================= CONFIGURAZIONE =================
WEB_PORT = 5000
MODEL_PATH = "../paligemma_offline"

STREAM_W = 1280
STREAM_H = 720
DASH_SCALE = 0.45  # Scala leggermente aumentata per riempire bene

# === PALETTE SORGENTI (CORNICI) ===
COL_DIRECTOR  = (0, 215, 255)   # ORO (BGR)
COL_ORCHESTRA = (255, 200, 0)   # TEAL (BGR)
COL_AUDIENCE  = (220, 220, 220) # SILVER (BGR)

# === PALETTE EMOZIONI (RGB per PIL) ===
# Questi colori si attivano solo quando viene rilevata l'emozione specifica
EMO_COLORS = {
    'joy':      (255, 220, 0),    # GIALLO SOLARE
    'anger':    (255, 40, 40),    # ROSSO FUOCO
    'surprise': (180, 0, 255),    # VIOLA ELETTRICO
    'default':  (240, 240, 240)   # BIANCO GHIACCIO (Per le altre)
}

SOURCES_CONFIG = [
    {"name": "DIRECTOR", "port": 9000, "color": COL_DIRECTOR},
    {"name": "ORCHESTRA", "port": 9001, "color": COL_ORCHESTRA},
    {"name": "AUDIENCE",  "port": 9002, "color": COL_AUDIENCE}
]

VALID_EMOTIONS = ['joy', 'anger', 'fear', 'disgust', 'surprise', 'sadness', 'boredom', 'neutral']
# ==================================================

app = Flask(__name__)

class SharedState:
    def __init__(self):
        self.frames = {src["name"]: None for src in SOURCES_CONFIG}
        self.emotions = {src["name"]: "WAITING" for src in SOURCES_CONFIG}
        self.output_frame = None
        self.lock = threading.Lock()
        self.running = True

shared_data = SharedState()

# --- HELPER FONTS & GRAFICA ---
def get_font(size, bold=False):
    try:
        if bold:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        else:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()

def draw_waveform(draw, x, y, w, h, color_rgb):
    """Disegna l'onda sonora con il colore dell'emozione"""
    bars = 15
    gap = 3
    bar_w = (w - (bars * gap)) / bars
    for i in range(bars):
        bar_h = random.randint(5, h)
        bx = x + (i * (bar_w + gap))
        by = y + (h - bar_h) / 2
        # Usa il colore passato come argomento (dinamico)
        draw.rectangle([bx, by, bx + bar_w, by + bar_h], fill=color_rgb)

def center_text(draw, text, font, total_width, y_pos, color):
    """Calcola la posizione X per centrare il testo perfettamente"""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x_pos = (total_width - text_w) / 2
    draw.text((x_pos, y_pos), text, font=font, fill=color)
    return x_pos + text_w # Ritorna la fine del testo per posizionare oggetti accanto

# --- WORKER VIDEO ---
def video_worker_pipe(source_conf):
    name = source_conf["name"]
    port = source_conf["port"]

    #srt_url = f"srt://0.0.0.0:{port}?mode=listener&recv_buffer_size=10000000&maxbw=5000000"

    
    srt_url = f"srt://0.0.0.0:{port}?mode=listener&latency=200000&recv_buffer_size=10000000"
    
    print(f"👀 [{name}] Video attivo.")

    """cmd = [
        'ffmpeg', '-re', '-thread_queue_size', '1024', '-loglevel', 'error',
        '-fflags', '+genpts+discardcorrupt', '-flags', 'low_delay',
        '-i', srt_url, '-r', '25',
        '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
    ]"""

    cmd = [
        'ffmpeg',
        '-hide_banner', '-loglevel', 'error', # Meno spam nella console
        '-thread_queue_size', '512',
        '-fflags', 'nobuffer',           # Niente buffer interno
        '-flags', 'low_delay',           # Modalità bassa latenza
        '-strict', 'experimental',
        
        # --- PARAMETRI DI ANALISI CORRETTI ---
        '-probesize', '256000',          # ~250KB (invece di 32 byte)
        '-analyzeduration', '200000',    # 0.2 secondi (invece di 0)
        
        # Input
        '-f', 'mpegts',                  # Forziamo il formato contenitore SRT standard
        '-i', srt_url,
        
        # --- FILTRI ---
        '-map', '0:v',                   # PRENDI SOLO IL VIDEO (Ignora Audio)
        '-r', '25',                      # Frame rate target
        
        # Output Pipe
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', '-'
    ]

    try:
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
    except:
        return
    frame_size = STREAM_W * STREAM_H * 3
    while shared_data.running:
        try:
            raw_image = pipe.stdout.read(frame_size)
            if len(raw_image) != frame_size:
                time.sleep(0.005)
                continue
            img = np.frombuffer(raw_image, dtype='uint8').reshape((STREAM_H, STREAM_W, 3))
            with shared_data.lock:
                shared_data.frames[name] = img
        except:
            pass
    pipe.terminate()

# --- WORKER AI ---
def analysis_worker(model, processor):
    print("🧠 [AI] Analisi Emozionale Attiva.")
    while shared_data.running:
        snapshot = {}
        with shared_data.lock:
            for k, v in shared_data.frames.items():
                if v is not None: snapshot[k] = v.copy()
        
        if not snapshot:
            time.sleep(0.1)
            continue

        for name, frame in snapshot.items():
            try:
                frame_small = cv2.resize(frame, (448, 448))
                img_pil = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_pil)
                
                prompt = f"<image>Detect emotion. Choose ONE: {VALID_EMOTIONS}."
                prompt1 = (
                    f" <image> What is the general emotion in this picture? "                    
                    f"You MUST choose ONLY ONE or TWO from: {VALID_EMOTIONS}."
                )
               
                inputs = processor(text=prompt1, images=img_pil, return_tensors="pt").to(model.device, model.dtype)
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                out = processor.decode(gen[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
                
                with shared_data.lock:
                    shared_data.emotions[name] = out
            except:
                pass
        time.sleep(0.05)

# --- WORKER GRAFICO (COMPOSITOR) ---
def compositor_worker():
    print("🎨 [GPU] Broadcast Compositor Avviato.")
    
    # Layout Spazioso
    sw = int(STREAM_W * DASH_SCALE)
    sh = int(STREAM_H * DASH_SCALE)
    
    # Header molto più alto per dare aria
    header_h = 160  
    gap = 40        # Spazio tra i video aumentato
    padding_side = 80 # Margini laterali

    # Larghezza totale calcolata sui contenuti
    total_w = (sw * 3) + (gap * 2) + (padding_side * 2)
    total_h = header_h + sh + 50 # +50 margine sotto

    # Font
    font_title = get_font(50, bold=True)      # Titolo enorme
    font_subtitle = get_font(18)              # Sottotitolo elegante
    font_cam_label = get_font(14, bold=True)  # Nome camera
    font_emo = get_font(22, bold=True)        # Emozione
    font_tech = get_font(10)

    while shared_data.running:
        # 1. Canvas Pulito (Nero Assoluto)
        dashboard = Image.new('RGB', (total_w, total_h), (0, 0, 0))
        draw = ImageDraw.Draw(dashboard, 'RGBA')
        
        # 2. HEADER CENTRALE (Minimalista)
        # Titolo
        title_end_x = center_text(draw, "M U S I C 4 D", font_title, total_w, 50, (255, 255, 255))
        
        # Sottotitolo
        center_text(draw, "GINEVRA OPERA // REAL-TIME EMOTION RECOGNITION", font_subtitle, total_w, 110, (100, 100, 100))
        
        # Live Dot (Puntino rosso pulsante accanto al titolo)
        # Lampeggia ogni secondo
        if int(time.time() * 2) % 2 == 0:
            # Calcoliamo posizione: a destra del titolo + 20px
            dot_x = title_end_x + 25
            dot_y = 65 # Circa a metà altezza del font titolo
            radius = 6
            draw.ellipse([dot_x, dot_y, dot_x + radius*2, dot_y + radius*2], fill=(255, 0, 0))

        # 3. COMPOSIZIONE VIDEO
        with shared_data.lock:
            cur_frames = shared_data.frames.copy()
            cur_emotions = shared_data.emotions.copy()

        for idx, conf in enumerate(SOURCES_CONFIG):
            name = conf["name"]
            # Colore Camera (Cornice)
            cam_col_bgr = conf["color"]
            cam_col_rgb = (cam_col_bgr[2], cam_col_bgr[1], cam_col_bgr[0]) 
            
            frame = cur_frames[name]
            emo_raw = cur_emotions[name].lower()
            
            # Colore Emozione (Testo e Onda)
            if emo_raw in EMO_COLORS:
                emo_col_rgb = EMO_COLORS[emo_raw]
            else:
                emo_col_rgb = EMO_COLORS['default']

            # Coordinate
            x = padding_side + (idx * (sw + gap))
            y = header_h

            if frame is None:
                # Placeholder minimal
                draw.rectangle([x, y, x+sw, y+sh], outline=(30,30,30), width=1)
                center_text(draw, "CONNECTING...", font_subtitle, (x*2 + sw), y + sh//2, (60,60,60)) 
            else:
                # Resize
                tile_cv = cv2.resize(frame, (sw, sh))
                tile_rgb = cv2.cvtColor(tile_cv, cv2.COLOR_BGR2RGB)
                tile_pil = Image.fromarray(tile_rgb)
                dashboard.paste(tile_pil, (x, y))
                
                # --- LAYOUT GRAFICO PULITO ---
                
                # 1. Cornice Camera Sottile (Solo angoli o intera? Intera è più ordinata qui)
                draw.rectangle([x, y, x+sw, y+sh], outline=cam_col_rgb, width=2)
                
                # 2. Etichetta Camera (In alto a SX, fuori dal video o dentro?)
                # Mettiamola dentro, in alto a sinistra, stile "tag"
                tag_w = 120
                tag_h = 24
                draw.rectangle([x, y, x+tag_w, y+tag_h], fill=(0,0,0)) # Sfondo nero pieno
                draw.rectangle([x, y, x+3, y+tag_h], fill=cam_col_rgb) # Accento colore camera
                draw.text((x + 10, y + 4), name, font=font_cam_label, fill=(200, 200, 200))

                # 3. Area Emozione (In basso, stile overlay cinema)
                # Sfondo sfumato scuro in basso
                overlay = Image.new('RGBA', (sw, 60), (0,0,0,0))
                dr_ov = ImageDraw.Draw(overlay)
                dr_ov.rectangle([0, 0, sw, 60], fill=(0,0,0,180)) # Nero semitrasparente
                dashboard.paste(overlay, (x, y + sh - 60), mask=overlay)

                # 4. Onda Sonora (Colorata in base all'emozione)
                draw_waveform(draw, x + 15, y + sh - 45, 80, 30, emo_col_rgb)

                # 5. Testo Emozione (Colorato in base all'emozione, allineato a destra)
                emo_text = emo_raw.upper()
                bbox = draw.textbbox((0, 0), emo_text, font=font_emo)
                txt_w = bbox[2] - bbox[0]
                
                draw.text((x + sw - txt_w - 15, y + sh - 42), emo_text, font=font_emo, fill=emo_col_rgb)
                
                # Etichetta "STATUS" piccola sopra l'emozione
                draw.text((x + sw - txt_w - 15, y + sh - 55), "STATUS", font=font_tech, fill=(100,100,100))

        # Conversione Finale
        final_np = np.array(dashboard)
        final_bgr = cv2.cvtColor(final_np, cv2.COLOR_RGB2BGR)

        with shared_data.lock:
            ret, buffer = cv2.imencode('.jpg', final_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ret:
                shared_data.output_frame = buffer.tobytes()
        
        time.sleep(0.04)

# --- SERVER WEB ---
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Music4D Control Room</title>
        <style>
            body { 
                background-color: #000; 
                margin: 0; padding: 0;
                display: flex; justify-content: center; align-items: center; 
                height: 100vh; width: 100vw; overflow: hidden;
            }
            img { 
                max-width: 98vw; max-height: 98vh;
                object-fit: contain;
                /* Nessun bordo, fluttua nel nero */
            }
        </style>
    </head>
    <body>
        <img src="/video_feed">
    </body>
    </html>
    """

def generate():
    while True:
        with shared_data.lock:
            frame = shared_data.output_frame
        if frame:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("--- MUSIC4D CLEAN DESIGN SERVER ---")
    
    try:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="auto", local_files_only=True 
        ).eval()
        processor = PaliGemmaProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        print("✅ AI Engine Ready.")
    except Exception as e:
        print(f"❌ Errore AI: {e}")
        exit()

    for conf in SOURCES_CONFIG:
        t = threading.Thread(target=video_worker_pipe, args=(conf,))
        t.daemon = True
        t.start()

    threading.Thread(target=analysis_worker, args=(model, processor), daemon=True).start()
    threading.Thread(target=compositor_worker, daemon=True).start()

    print(f"🚀 DASHBOARD: http://0.0.0.0:{WEB_PORT}")
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True)