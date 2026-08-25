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
from flask import Flask, Response
from transformers import AutoModel, AutoTokenizer
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

# ================= CONFIGURAZIONE =================
WEB_PORT = 5000
MODEL_PATH = "openbmb/MiniCPM-V-4"

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
def analysis_worker(model, tokenizer):
    print("🧠 [AI] Analisi Emozionale Attiva con MiniCPM-V-4.")
    history_emo = {src["name"]: deque(maxlen=5) for src in SOURCES_CONFIG}
    
    while shared_data.running:
        snapshot = {}
        with shared_data.lock:
            for k, v in shared_data.frames.items():
                if v is not None: snapshot[k] = v.copy()
        
        if not snapshot:
            time.sleep(0.1); continue

        for name, frame in snapshot.items():
            try:
                # Conversione per MiniCPM (richiede immagine RGB)
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')

                # Costruzione del messaggio secondo le specifiche MiniCPM-V
                msgs = [{'role': 'user', 'content': [
                    img_pil,
                    f"What is the general emotion in this picture? You MUST choose ONLY ONE from: {VALID_EMOTIONS}. Output just the word."
                ]}]

                prompt_text = (
                    "You are an expert emotion recognition system analyzing a live orchestral performance. "
                    "Observe the facial expressions and body language of the people in the image (conductor, musicians, or audience). "
                    f"Classify the dominant collective emotion choosing EXACTLY ONE word from this exact list: {VALID_EMOTIONS}. "
                    "Respond with just the single word. No punctuation, no explanations, no introductory text."
                )

                msgs1 = [{'role': 'user', 'content': [
                    img_pil,
                    prompt_text
                ]}]

               

                # Inferenza
                with torch.inference_mode():
                    res = model.chat(
                        image=None,
                        msgs=msgs1,
                        tokenizer=tokenizer,
                        sampling=False, 
                        #temperature=0.7
                    )
                
                # Estrazione e pulizia del testo
                out = res.lower().strip()
                out_clean = ''.join(filter(str.isalpha, out))
                
                # Default fallback se il modello fallisce nel seguire l'istruzione esatta
                if out_clean not in VALID_EMOTIONS:
                    out_clean = "neutral"

                history_emo[name].append(out_clean)
                most_common_emo = Counter(history_emo[name]).most_common(1)[0][0]
                
                with shared_data.lock: 
                    shared_data.emotions[name] = most_common_emo
            except Exception as e: 
                # print(f"⚠️ Errore AI su {name}: {e}")
                pass
        
        time.sleep(0.08)

# --- WORKER GRAFICO  ---
def compositor_worker():
    print("🎨 [GPU] Broadcast Compositor Avviato.")
    
    CANVAS_W = 1920
    CANVAS_H = 1080
    
    # LAYOUT MAXIMIZED
    gap = 20           
    margin_side = 30   
    
    avail_w = CANVAS_W - (margin_side * 2) - (gap * 2)
    single_w = int(avail_w / 3) 
    single_h = int(single_w * (9/16)) 
    
    video_y = int((CANVAS_H - single_h) / 2) - 20 

    # Fonts
    font_title = get_font(56, bold=True)
    font_sub   = get_font(20)
    font_label = get_font(16, bold=True)
    font_emo   = get_font(34, bold=True)
    font_tech  = get_font(12)

    # Base Sfondo
    base_bg = np.full((CANVAS_H, CANVAS_W, 3), COL_BG_BLACK, dtype=np.uint8)

    while shared_data.running:
        final_canvas = base_bg.copy()
        
        with shared_data.lock:
            cur_frames = shared_data.frames.copy()
            cur_emotions = shared_data.emotions.copy()

        cv2.line(final_canvas, (0, video_y - 60), (CANVAS_W, video_y - 60), (30,30,30), 1)        
        cv2.line(final_canvas, (0, video_y + single_h + 110), (CANVAS_W, video_y + single_h + 110), (30,30,30), 1)

        video_coords = []
        
        # 1. VIDEO
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

        # 2. OVERLAY GRAFICO
        pil_img = Image.fromarray(cv2.cvtColor(final_canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, 'RGBA')

        # --- HEADER ---
        title_text = "M U S I C 4 D"
        sub_text = "GENEVA ASSEMBLY HALL // REAL-TIME EMOTION RECOGNITION"
        
        # Posizionamento Titolo 
        title_y = 65 
        bbox_t = draw.textbbox((0, 0), title_text, font=font_title)
        title_w = bbox_t[2] - bbox_t[0]
        title_h = bbox_t[3] - bbox_t[1]
        
        title_x = (CANVAS_W - title_w) / 2
        
        draw.text((title_x, title_y), title_text, font=font_title, fill=TEXT_WHITE)        
        
        dot_x = title_x + title_w + 20
        dot_y = title_y + (title_h / 2) + 5
        dot_r = 6 
        
        if int(time.time() * 2) % 2 == 0:
            draw.ellipse((dot_x, dot_y - dot_r, dot_x + dot_r*2, dot_y + dot_r), fill=ACCENT_RED)
        
        # Sottotitolo
        bbox_s = draw.textbbox((0, 0), sub_text, font=font_sub)
        sub_w = bbox_s[2] - bbox_s[0]
        draw.text(((CANVAS_W - sub_w) / 2, title_y + 70), sub_text, font=font_sub, fill=TEXT_GREY)

        # --- CARDS VIDEO ---
        for i, (x, y, name, conf) in enumerate(video_coords):
            bgr = conf["color"]
            rgb = (bgr[2], bgr[1], bgr[0]) 
            
            emo_raw = cur_emotions[name].lower()
            emo_rgb = EMO_COLORS.get(emo_raw, EMO_COLORS['default'])

            # Cornice
            draw.rectangle([x, y, x+single_w, y+single_h], outline=rgb, width=2)
            
            # Etichetta Nome
            draw.rectangle([x, y, x + 120, y + 30], fill=(0,0,0, 220)) 
            draw.rectangle([x, y, x + 4, y + 30], fill=rgb)            
            draw.text((x + 15, y + 5), name, font=font_label, fill=TEXT_WHITE)

            # --- DATA PANEL ---
            panel_y = y + single_h + 10
            draw.rectangle([x, panel_y, x+single_w, panel_y + 80], fill=(10,10,10))            
            draw_waveform(draw, x + 10, panel_y + 25, 120, 30, emo_rgb)
            
            # Emozione
            emo_text = emo_raw.upper()
            if emo_text == "WAITING": emo_text = "..."
            
            bbox = draw.textbbox((0,0), emo_text, font=font_emo)
            ew = bbox[2] - bbox[0]
            text_x = x + single_w - ew - 10
            
            # Glow
            for off in range(1, 3):
                draw.text((text_x + off, panel_y + 10 + off), emo_text, font=font_emo, fill=(emo_rgb[0], emo_rgb[1], emo_rgb[2], 50))
            
            draw.text((text_x, panel_y + 10), emo_text, font=font_emo, fill=emo_rgb)
            
            # Label STATUS DETECTED
            draw.text((text_x, panel_y + 60), "DETECTED", font=font_tech, fill=(80,80,80))

        # OUTPUT
        final_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with shared_data.lock:
            ret, buffer = cv2.imencode('.jpg', final_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ret: shared_data.output_frame = buffer.tobytes()
        
        time.sleep(0.04)

# --- SERVER WEB ---
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

if __name__ == "__main__":
    print("--- MUSIC4D SERVER ---")
    try:
        # Caricamento del nuovo modello OpenBMB MiniCPM-V-4
        model = AutoModel.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True, 
            attn_implementation='sdpa', 
            torch_dtype=torch.bfloat16
        ).to(device="cuda").eval()
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("✅ AI Ready.")
    except Exception as e: 
        print(f"❌ Error: {e}")
        exit()

    for conf in SOURCES_CONFIG:
        threading.Thread(target=video_worker_pipe, args=(conf,), daemon=True).start()

    threading.Thread(target=analysis_worker, args=(model, tokenizer), daemon=True).start()
    threading.Thread(target=compositor_worker, daemon=True).start()

    print(f"🚀 DASHBOARD: http://0.0.0.0:{WEB_PORT}")
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True)