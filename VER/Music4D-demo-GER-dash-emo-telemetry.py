import sys
import time
import threading
import numpy as np
import os
import subprocess
import datetime
import random
import atexit
import signal
from collections import deque, Counter
from flask import Flask, Response, jsonify
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

# ================= CONFIGURAZIONE =================
WEB_PORT = 5000
MODEL_PATH = "../paligemma_offline"

# Risoluzione Stream Input
STREAM_W = 1280
STREAM_H = 720

# === PALETTE COLORI ===
COL_BG_BLACK   = (0, 0, 0)   
COL_FRAME_WHITE = (255, 255, 255) 
COL_DIRECTOR   = COL_FRAME_WHITE
COL_ORCHESTRA  = COL_FRAME_WHITE
COL_AUDIENCE   = COL_FRAME_WHITE

TEXT_WHITE     = (255, 255, 255)
TEXT_GREY      = (120, 120, 120)
ACCENT_RED     = (255, 0, 0)      

# Colori Emozioni (RGB per PIL)
EMO_COLORS = {
    'joy':      (255, 220, 0),    # Giallo
    'anger':    (255, 40, 40),    # Rosso
    'surprise': (200, 100, 255),  # Viola chiaro
    'sadness':  (80, 140, 255),   # Blu
    'fear':     (255, 140, 0),    # Arancio
    'default':  (255, 255, 255)   # Bianco
}

# === CONFIGURAZIONE PORTE VIDEOCAMERE ===
SOURCES_CONFIG = [
    {"name": "DIRECTOR",  "port": 9000, "color": COL_DIRECTOR},
    {"name": "ORCHESTRA", "port": 9001, "color": COL_ORCHESTRA},
    {"name": "AUDIENCE",  "port": 9002, "color": COL_AUDIENCE}
]

# === LISTA EMOZIONI ===
VALID_EMOTIONS = ['joy', 'anger', 'fear', 'disgust', 'surprise', 'sadness', 'boredom', 'neutral']
# ==================================================

app = Flask(__name__)

# --- GESTIONE PROCESSI ---
ffmpeg_processes = []

def cleanup_processes():
    if ffmpeg_processes:
        print("\n🧹 [SYSTEM] Pulizia processi FFmpeg...")
        for p in ffmpeg_processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    p.kill()

atexit.register(cleanup_processes)

# --- STATO CONDIVISO ---
class SharedState:
    def __init__(self):
        self.frames = {src["name"]: None for src in SOURCES_CONFIG}
        self.emotions = {src["name"]: "WAITING" for src in SOURCES_CONFIG}
        self.output_frame = None
        self.lock = threading.Lock()
        self.running = True

shared_data = SharedState()

# --- HELPER FONTS ---
def get_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]
    for path in candidates:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, size)
            except: continue
    return ImageFont.load_default()

def draw_waveform(draw, x, y, w, h, color_rgb):
    bars = 16
    gap = 3
    bar_w = (w - (bars * gap)) / bars
    t = time.time() * 12
    for i in range(bars):
        noise = (np.sin(t + i*0.5) + 1) / 2 
        bar_h = 4 + (noise * (h - 4))
        bx = x + (i * (bar_w + gap))
        by = y + (h - bar_h) / 2
        draw.rectangle([bx, by, bx + bar_w, by + bar_h], fill=color_rgb)

# --- WORKER VIDEO ---
def video_worker_pipe(source_conf):
    name = source_conf["name"]
    port = source_conf["port"]
    
    srt_url = f"srt://0.0.0.0:{port}?mode=listener&recv_buffer_size=10000000"

    print(f"👀 [{name}] Video attivo su porta {port}...")
    
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-thread_queue_size', '512',
        '-fflags', 'nobuffer', '-flags', 'low_delay', '-strict', 'experimental',
        '-probesize', '256000', '-analyzeduration', '200000',
        '-f', 'mpegts', '-i', srt_url,
        '-map', '0:v', '-r', '25',
        '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
    ]
    
    try:
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        ffmpeg_processes.append(pipe)
    except Exception as e:
        print(f"❌ Errore avvio FFmpeg [{name}]: {e}")
        return

    frame_size = STREAM_W * STREAM_H * 3
    
    while shared_data.running:
        try:
            raw_image = pipe.stdout.read(frame_size)
            if len(raw_image) != frame_size:
                if pipe.poll() is not None: break
                time.sleep(0.002)
                continue
            img = np.frombuffer(raw_image, dtype='uint8').reshape((STREAM_H, STREAM_W, 3))
            with shared_data.lock: shared_data.frames[name] = img
        except: break
    pipe.terminate()

# --- WORKER AI ---
def analysis_worker(model, processor):
    print("🧠 [AI] Analisi Emozionale Attiva.")
    history = {src["name"]: deque(maxlen=5) for src in SOURCES_CONFIG}
    
    while shared_data.running:
        snapshot = {}
        with shared_data.lock:
            for k, v in shared_data.frames.items():
                if v is not None: snapshot[k] = v.copy()
        
        if not snapshot:
            time.sleep(0.1); continue

        for name, frame in snapshot.items():
            try:
                frame_small = cv2.resize(frame, (448, 448))
                img_pil = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))

                prompt1 = (
                    f" <image> What is the general emotion in this picture? "                    
                    f"You MUST choose ONLY ONE  from: {VALID_EMOTIONS}."
                )

                inputs = processor(text=prompt1, images=img_pil, return_tensors="pt").to(model.device, model.dtype)
                
                with torch.inference_mode():
                    gen = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                
                out = processor.decode(gen[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().lower()
                out_clean = ''.join(filter(str.isalpha, out))
                
                history[name].append(out_clean)
                most_common_emo = Counter(history[name]).most_common(1)[0][0]
                
                with shared_data.lock: shared_data.emotions[name] = most_common_emo
            except: pass
        time.sleep(0.08)

# --- WORKER GRAFICO AGGIORNATO ---
def compositor_worker():
    print("🎨 [GPU] Broadcast Compositor Avviato.")
    
    # --- CARICAMENTO LOGO ---
    try:
        # Carichiamo il logo e convertiamo in RGBA per gestire la trasparenza
        logo_img = Image.open("logo.png").convert("RGBA")
    except Exception as e:
        print(f"⚠️ Errore caricamento logo: {e}")
        logo_img = None

    CANVAS_W = 1920
    CANVAS_H = 1080
    
    gap = 20           
    margin_side = 30   
    
    avail_w = CANVAS_W - (margin_side * 2) - (gap * 2)
    single_w = int(avail_w / 3) 
    single_h = int(single_w * (9/16)) 
    
    video_y = int((CANVAS_H - single_h) / 2) - 20 

    font_title = get_font(56, bold=True)
    font_sub   = get_font(20)
    font_label = get_font(16, bold=True)
    font_emo   = get_font(34, bold=True)
    font_tech  = get_font(12)

    base_bg = np.full((CANVAS_H, CANVAS_W, 3), COL_BG_BLACK, dtype=np.uint8)

    while shared_data.running:
        final_canvas = base_bg.copy()
        
        with shared_data.lock:
            cur_frames = shared_data.frames.copy()
            cur_emotions = shared_data.emotions.copy()

        cv2.line(final_canvas, (0, video_y - 60), (CANVAS_W, video_y - 60), (30,30,30), 1)        
        cv2.line(final_canvas, (0, video_y + single_h + 110), (CANVAS_W, video_y + single_h + 110), (30,30,30), 1)

        video_coords = []
        
        for idx, conf in enumerate(SOURCES_CONFIG):
            name = conf["name"]
            x = margin_side + (idx * (single_w + gap))
            y = video_y
            
            frame = cur_frames[name]
            
            if frame is not None:
                small_f = cv2.resize(frame, (single_w, single_h))
                final_canvas[y:y+single_h, x:x+single_w] = small_f
            else:
                cv2.rectangle(final_canvas, (x, y), (x+single_w, y+single_h), (15, 15, 15), -1)
                
            video_coords.append((x, y, name, conf))

        pil_img = Image.fromarray(cv2.cvtColor(final_canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        # --- LOGICA TITOLO E LOGO ---
        title_text = "M U S I C 4 D"
        title_y = 65 
        
        # Calcolo dimensioni testo
        bbox_t = draw.textbbox((0, 0), title_text, font=font_title)
        title_w = bbox_t[2] - bbox_t[0]
        title_h = bbox_t[3] - bbox_t[1]
        
        logo_spacing = 25  # Spazio tra logo e testo
        total_header_w = title_w
        
        # Ridimensionamento dinamico del logo in base all'altezza del font
        if logo_img:
            logo_h = int(title_h * 1.5) # Un po' più alto del font per visibilità
            aspect_ratio = logo_img.width / logo_img.height
            logo_w = int(logo_h * aspect_ratio)
            current_logo = logo_img.resize((logo_w, logo_h), Image.Resampling.LANCZOS)
            total_header_w += (logo_w + logo_spacing)
        
        # Calcolo coordinata X di partenza per centrare l'intero blocco (Logo + Testo)
        start_x = (CANVAS_W - total_header_w) / 2
        
        # Disegno il Logo
        if logo_img:
            logo_y = title_y + (title_h // 2) - (logo_h // 2) + 5 # Centratura verticale rispetto al testo
            pil_img.paste(current_logo, (int(start_x), int(logo_y)), current_logo)
            title_x = start_x + logo_w + logo_spacing
        else:
            title_x = start_x

        # Disegno il Testo
        draw.text((title_x, title_y), title_text, font=font_title, fill=TEXT_WHITE)        
        
        # Puntino rosso (REC) a destra del testo
        dot_x = title_x + title_w + 20
        dot_y = title_y + (title_h / 2) + 5
        dot_r = 6 
        
        if int(time.time() * 2) % 2 == 0:
            draw.ellipse((dot_x, dot_y - dot_r, dot_x + dot_r*2, dot_y + dot_r), fill=ACCENT_RED)
        
        # Sottotitolo centrato (rispetto al canvas)
        sub_text = "TAU // University of Calabria "
        bbox_s = draw.textbbox((0, 0), sub_text, font=font_sub)
        sub_w = bbox_s[2] - bbox_s[0]
        draw.text(((CANVAS_W - sub_w) / 2, title_y + 70), sub_text, font=font_sub, fill=TEXT_GREY)

        # --- RESTO DEL CODICE (Video Panels, Waveforms, etc.) ---
        for i, (x, y, name, conf) in enumerate(video_coords):
            bgr = conf["color"]
            rgb = (bgr[2], bgr[1], bgr[0]) 
            
            emo_raw = cur_emotions[name].lower()
            emo_rgb = EMO_COLORS.get(emo_raw, EMO_COLORS['default'])

            draw.rectangle([x, y, x+single_w, y+single_h], outline=rgb, width=2)
            
            draw.rectangle([x, y, x + 120, y + 30], fill=(0,0,0, 220)) 
            draw.rectangle([x, y, x + 4, y + 30], fill=rgb)            
            draw.text((x + 15, y + 5), name, font=font_label, fill=TEXT_WHITE)

            panel_y = y + single_h + 10
            draw.rectangle([x, panel_y, x+single_w, panel_y + 80], fill=(10,10,10))            
            draw_waveform(draw, x + 10, panel_y + 25, 120, 30, emo_rgb)
            
            emo_text = emo_raw.upper()
            if emo_text == "WAITING": emo_text = "..."
            
            bbox = draw.textbbox((0,0), emo_text, font=font_emo)
            ew = bbox[2] - bbox[0]
            text_x = x + single_w - ew - 10
            
            for off in range(1, 3):
                draw.text((text_x + off, panel_y + 10 + off), emo_text, font=font_emo, fill=(emo_rgb[0], emo_rgb[1], emo_rgb[2], 50))
            
            draw.text((text_x, panel_y + 10), emo_text, font=font_emo, fill=emo_rgb)
            draw.text((text_x, panel_y + 60), "DETECTED", font=font_tech, fill=(80,80,80))

        final_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with shared_data.lock:
            ret, buffer = cv2.imencode('.jpg', final_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ret: shared_data.output_frame = buffer.tobytes()
        
        time.sleep(0.04)

# --- SERVER WEB E API ---
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Music4D Monitor</title>
        <style>
            body { 
                background-color: #000000; 
                margin: 0; padding: 0;
                height: 100vh; width: 100vw; 
                display: flex; justify-content: center; align-items: center;
                overflow: hidden;
            }
            img { 
                max-width: 100%; 
                max-height: 100%;
                object-fit: contain; 
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
        with shared_data.lock: frame = shared_data.output_frame
        if frame: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    """Endpoint API per il team dei robot per fare il PULL delle emozioni in tempo reale."""
    with shared_data.lock:
        cur_emotions = shared_data.emotions.copy()
        
    valid_emos = [e for e in cur_emotions.values() if e != "WAITING"]
    
    if valid_emos:
        dominant = Counter(valid_emos).most_common(1)[0][0]
    else:
        dominant = "neutral"
        
    color_rgb = list(EMO_COLORS.get(dominant, EMO_COLORS['default'])) 
    color_hex = f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"
    
    payload = {
        "timestamp": time.time(),
        "entities": cur_emotions,
        "dominant_emotion": dominant,
        "dominant_color_rgb": color_rgb,
        "dominant_color_hex": color_hex
    }
    
    return jsonify(payload)

if __name__ == "__main__":
    print("--- MUSIC4D SERVER ---")
    try:
        model = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="auto", local_files_only=True).eval()
        processor = PaliGemmaProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        print("✅ AI Ready.")
    except Exception as e: 
        print(f"❌ Error: {e}")
        exit()

    for conf in SOURCES_CONFIG:
        threading.Thread(target=video_worker_pipe, args=(conf,), daemon=True).start()

    threading.Thread(target=analysis_worker, args=(model, processor), daemon=True).start()
    threading.Thread(target=compositor_worker, daemon=True).start()

    print(f"🚀 DASHBOARD: http://0.0.0.0:{WEB_PORT}")
    print(f"📡 API EMOZIONI: http://0.0.0.0:{WEB_PORT}/api/emotions")
    
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True)