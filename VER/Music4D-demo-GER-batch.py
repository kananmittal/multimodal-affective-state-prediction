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
EMO_COLORS = {
    'joy':      (255, 220, 0),    # GIALLO SOLARE
    'anger':    (255, 40, 40),    # ROSSO FUOCO
    'surprise': (180, 0, 255),    # VIOLA ELETTRICO
    'default':  (240, 240, 240)   # BIANCO GHIACCIO
}

SOURCES_CONFIG = [
    {"name": "DIRECTOR", "port": 9000, "color": COL_DIRECTOR},
    {"name": "ORCHESTRA", "port": 9001, "color": COL_ORCHESTRA},
    {"name": "AUDIENCE",  "port": 9002, "color": COL_AUDIENCE}
]

VALID_EMOTIONS = ['joy', 'anger', 'fear', 'disgust', 'surprise', 'sadness', 'boredom', 'neutral']

# === NUOVA CLASSE PER FRAME TIMESTAMPED ===
class TimestampedFrame:
    def __init__(self, frame, timestamp):
        self.frame = frame.copy() if frame is not None else None
        self.timestamp = timestamp
        self.emotion = "WAITING"
        self.emotion_timestamp = 0
        self.last_analysis = 0
        self.last_emotion = ""
        self.emotion_streak = 0  # Per caching emozioni stabili
# ==================================================

app = Flask(__name__)

class SharedState:
    def __init__(self):
        self.frames = {src["name"]: None for src in SOURCES_CONFIG}
        self.emotions = {src["name"]: "WAITING" for src in SOURCES_CONFIG}
        self.output_frame = None
        self.lock = threading.Lock()
        self.running = True
        # Per statistiche
        self.inference_stats = {
            "times": [],
            "count": 0,
            "fps": 0.0,
            "batch_sizes": [],
            "last_batch_time": 0
        }

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
    bars = 15
    gap = 3
    bar_w = (w - (bars * gap)) / bars
    for i in range(bars):
        bar_h = random.randint(5, h)
        bx = x + (i * (bar_w + gap))
        by = y + (h - bar_h) / 2
        draw.rectangle([bx, by, bx + bar_w, by + bar_h], fill=color_rgb)

def center_text(draw, text, font, total_width, y_pos, color):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x_pos = (total_width - text_w) / 2
    draw.text((x_pos, y_pos), text, font=font, fill=color)
    return x_pos + text_w

# --- WORKER VIDEO (RIMANE SIMILE) ---
def video_worker_pipe(source_conf):
    name = source_conf["name"]
    port = source_conf["port"]

    srt_url = f"srt://0.0.0.0:{port}?mode=listener&latency=200000&recv_buffer_size=10000000"

    print(f"👀 [{name}] Video attivo.")
    
    cmd = [
        'ffmpeg', '-re', '-thread_queue_size', '512',  # Ridotto per latenza
        '-loglevel', 'error',
        '-fflags', '+nobuffer+fastseek',
        '-flags', 'low_delay',
        '-i', srt_url, '-r', '25',
        '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'
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
                if shared_data.frames[name] is None:
                    shared_data.frames[name] = TimestampedFrame(img, time.time())
                else:
                    shared_data.frames[name].frame = img.copy()
                    shared_data.frames[name].timestamp = time.time()
        except Exception as e:
            # print(f"❌ [{name}] Video error: {e}")
            time.sleep(0.1)
    pipe.terminate()

# --- WORKER AI CON BATCH PROCESSING ---
def analysis_worker(model, processor):
    print("🧠 [AI] Analisi Emozionale BATCH Attiva.")
    
    # Configurazioni ottimizzate per batch
    min_analysis_interval = 0.25  # 4 FPS AI per sorgente
    image_size = 384  # Ridotto da 448 per batch più veloce
    max_new_tokens = 5  # Ridotto da 8
    prompt = "<image>emotion: joy,anger,fear,disgust,surprise,sadness,boredom,neutral"
    
    # Statistiche
    last_log_time = time.time()
    batch_times = []
    
    # Cache per emozioni stabili
    emotion_cache_ttl = 2.0  # Secondi di cache se emozione stabile
    
    while shared_data.running:
        current_time = time.time()
        
        # 1. RACCOLTA BATCH - Prende TUTTE le sorgenti pronte
        batch_frames = []
        batch_sources = []
        batch_objects = []
        
        with shared_data.lock:
            for name, frame_obj in shared_data.frames.items():
                if frame_obj is None or frame_obj.frame is None:
                    continue
                
                # Calcola intervallo dinamico basato su stabilità emozione
                if frame_obj.emotion_streak > 3:  # Emozione stabile da 3 frame
                    effective_interval = min_analysis_interval * 1.5
                else:
                    effective_interval = min_analysis_interval
                
                # Controlla se è il momento di analizzare
                time_since_last = current_time - frame_obj.last_analysis
                if time_since_last > effective_interval:
                    batch_frames.append(frame_obj.frame)
                    batch_sources.append(name)
                    batch_objects.append(frame_obj)
                    frame_obj.last_analysis = current_time
        
        # 2. BATCH INFERENCE (se ci sono frame da processare)
        if batch_frames:
            try:
                start_inference = time.perf_counter()
                
                # PREPARAZIONE BATCH IMMAGINI
                processed_images = []
                for frame in batch_frames:
                    # Resize ottimizzato
                    frame_small = cv2.resize(frame, (image_size, image_size))
                    img_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    processed_images.append(img_pil)
                
                # BATCH PROCESSING (il cuore dell'ottimizzazione)
                # Crea un batch con lo stesso prompt per tutte le immagini
                inputs = processor(
                    text=[prompt] * len(processed_images),  # Prompt identico per tutti
                    images=processed_images,
                    return_tensors="pt",
                    padding=True,  # Importante per batch
                    truncation=True
                ).to(model.device, model.dtype)
                
                # INFERENZA BATCH
                with torch.inference_mode():
                    # Temperature 0 per output deterministico
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
                
                # 3. PROCESSING RISULTATI BATCH
                batch_results = []
                for i in range(len(batch_sources)):
                    # Decodifica solo la parte generata (dopo i token di input)
                    input_length = inputs.input_ids.shape[-1]
                    output_ids = gen[i][input_length:]
                    out_text = processor.decode(output_ids, skip_special_tokens=True).strip().lower()
                    
                    # Validazione emozione
                    detected_emotion = "neutral"
                    for emotion in VALID_EMOTIONS:
                        if emotion in out_text:
                            detected_emotion = emotion
                            break
                    
                    batch_results.append({
                        'source': batch_sources[i],
                        'emotion': detected_emotion,
                        'frame_obj': batch_objects[i],
                        'raw_output': out_text
                    })
                
                # 4. APPLICA RISULTATI BATCH (con aggiornamento cache)
                with shared_data.lock:
                    for result in batch_results:
                        name = result['source']
                        detected_emotion = result['emotion']
                        frame_obj = result['frame_obj']
                        
                        # Aggiorna contatore stabilità emozione
                        if detected_emotion == frame_obj.last_emotion:
                            frame_obj.emotion_streak += 1
                        else:
                            frame_obj.emotion_streak = 0
                            frame_obj.last_emotion = detected_emotion
                        
                        # Aggiorna stato
                        frame_obj.emotion = detected_emotion
                        frame_obj.emotion_timestamp = current_time
                        shared_data.emotions[name] = detected_emotion.upper()
                
                # 5. CALCOLA STATISTICHE BATCH
                inference_time = time.perf_counter() - start_inference
                time_per_frame = inference_time / len(batch_frames)
                
                batch_times.append(inference_time)
                batch_times = batch_times[-50:]  # Mantieni ultimi 50 batch
                
                with shared_data.lock:
                    shared_data.inference_stats["times"].append(time_per_frame)
                    shared_data.inference_stats["times"] = shared_data.inference_stats["times"][-100:]
                    shared_data.inference_stats["batch_sizes"].append(len(batch_frames))
                    shared_data.inference_stats["batch_sizes"] = shared_data.inference_stats["batch_sizes"][-50:]
                    shared_data.inference_stats["count"] += len(batch_frames)
                    shared_data.inference_stats["last_batch_time"] = current_time
                    
                    # Calcola FPS efficace
                    if shared_data.inference_stats["times"]:
                        avg_time = sum(shared_data.inference_stats["times"]) / len(shared_data.inference_stats["times"])
                        shared_data.inference_stats["fps"] = 1.0 / avg_time if avg_time > 0 else 0
                
                # Log batch performance
                if len(batch_frames) > 1:  # Solo se vero batch
                    print(f"🔁 [AI] Batch {len(batch_frames)} frames: {inference_time*1000:.1f}ms total, {time_per_frame*1000:.1f}ms per frame")
                
            except Exception as e:
                print(f"⚠️ [AI] Batch error: {e}")
                # Fallback: processa singolarmente in caso di errore batch
                for i, name in enumerate(batch_sources):
                    try:
                        # Processa frame singolo come fallback
                        frame_obj = batch_objects[i]
                        frame_small = cv2.resize(frame_obj.frame, (image_size, image_size))
                        img_pil = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(img_pil)
                        
                        inputs = processor(text=prompt, images=img_pil, return_tensors="pt").to(model.device, model.dtype)
                        
                        with torch.inference_mode():
                            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                        
                        out = processor.decode(gen[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().lower()
                        
                        detected_emotion = "neutral"
                        for emotion in VALID_EMOTIONS:
                            if emotion in out:
                                detected_emotion = emotion
                                break
                        
                        with shared_data.lock:
                            frame_obj.emotion = detected_emotion
                            frame_obj.emotion_timestamp = time.time()
                            shared_data.emotions[name] = detected_emotion.upper()
                            
                    except:
                        pass
        
        # 6. LOG STATISTICHE PERIODICHE
        if current_time - last_log_time > 10:
            with shared_data.lock:
                if shared_data.inference_stats["times"]:
                    avg_time = sum(shared_data.inference_stats["times"]) / len(shared_data.inference_stats["times"])
                    fps = shared_data.inference_stats["fps"]
                    
                    if shared_data.inference_stats["batch_sizes"]:
                        avg_batch = sum(shared_data.inference_stats["batch_sizes"]) / len(shared_data.inference_stats["batch_sizes"])
                    else:
                        avg_batch = 0
                    
                    print(f"📊 [AI] Stats: {avg_time*1000:.1f}ms/frame, {fps:.1f} FPS, "
                          f"Avg batch: {avg_batch:.1f} frames, Total: {shared_data.inference_stats['count']}")
            
            last_log_time = current_time
            batch_times = []
        
        # 7. SLEEP DINAMICO BASATO SU CARICO
        # Se abbiamo processato un batch grande, possiamo dormire di più
        sleep_base = 0.03  # 30ms base
        if batch_frames:
            # Aggiusta sleep in base alla dimensione del batch
            sleep_time = sleep_base + (len(batch_frames) * 0.01)
        else:
            sleep_time = sleep_base
        
        time.sleep(min(sleep_time, 0.1))  # Max 100ms

# --- WORKER GRAFICO (AGGIORNATO PER STATS BATCH) ---
def compositor_worker():
    print("🎨 [GPU] Broadcast Compositor Avviato.")
    
    # Layout
    sw = int(STREAM_W * DASH_SCALE)
    sh = int(STREAM_H * DASH_SCALE)
    header_h = 160  
    gap = 40        
    padding_side = 80
    total_w = (sw * 3) + (gap * 2) + (padding_side * 2)
    total_h = header_h + sh + 50
    
    # Font
    font_title = get_font(50, bold=True)
    font_subtitle = get_font(18)
    font_cam_label = get_font(14, bold=True)
    font_emo = get_font(22, bold=True)
    font_tech = get_font(10)
    font_stats = get_font(12)
    
    last_stats_update = 0
    display_stats = "AI: -- ms (batch: --) | Sync: -- ms"
    
    while shared_data.running:
        current_time = time.time()
        
        # Aggiorna statistiche ogni secondo
        if current_time - last_stats_update > 1:
            with shared_data.lock:
                if shared_data.inference_stats["times"]:
                    avg_inference = sum(shared_data.inference_stats["times"]) / len(shared_data.inference_stats["times"])
                    
                    if shared_data.inference_stats["batch_sizes"]:
                        avg_batch = sum(shared_data.inference_stats["batch_sizes"]) / len(shared_data.inference_stats["batch_sizes"])
                        batch_str = f"batch: {avg_batch:.1f}"
                    else:
                        batch_str = "batch: --"
                    
                    display_stats = f"AI: {avg_inference*1000:.0f}ms ({batch_str}) | Sync: <400ms"
            last_stats_update = current_time
        
        # 1. Canvas
        dashboard = Image.new('RGB', (total_w, total_h), (0, 0, 0))
        draw = ImageDraw.Draw(dashboard, 'RGBA')
        
        # 2. HEADER con statistiche batch
        title_end_x = center_text(draw, "M U S I C 4 D", font_title, total_w, 50, (255, 255, 255))
        center_text(draw, f"GINEVRA OPERA // {display_stats}", font_subtitle, total_w, 110, (100, 100, 100))
        
        # Live Dot
        if int(time.time() * 2) % 2 == 0:
            dot_x = title_end_x + 25
            dot_y = 65
            radius = 6
            draw.ellipse([dot_x, dot_y, dot_x + radius*2, dot_y + radius*2], fill=(255, 0, 0))
        
        # 3. COMPOSIZIONE VIDEO
        with shared_data.lock:
            cur_frames = shared_data.frames.copy()
        
        for idx, conf in enumerate(SOURCES_CONFIG):
            name = conf["name"]
            cam_col_bgr = conf["color"]
            cam_col_rgb = (cam_col_bgr[2], cam_col_bgr[1], cam_col_bgr[0])
            
            frame_obj = cur_frames[name]
            
            # Coordinate
            x = padding_side + (idx * (sw + gap))
            y = header_h
            
            if frame_obj is None or frame_obj.frame is None:
                # Placeholder
                draw.rectangle([x, y, x+sw, y+sh], outline=(30,30,30), width=1)
                center_text(draw, "CONNECTING...", font_subtitle, (x*2 + sw), y + sh//2, (60,60,60))
                continue
            
            # Frame principale
            tile_cv = cv2.resize(frame_obj.frame, (sw, sh))
            tile_rgb = cv2.cvtColor(tile_cv, cv2.COLOR_BGR2RGB)
            tile_pil = Image.fromarray(tile_rgb)
            dashboard.paste(tile_pil, (x, y))
            
            # Colore emozione
            emo_raw = frame_obj.emotion.lower()
            if emo_raw in EMO_COLORS:
                emo_col_rgb = EMO_COLORS[emo_raw]
            else:
                emo_col_rgb = EMO_COLORS['default']
            
            # Calcola ritardo sincronizzazione
            sync_delay = current_time - frame_obj.emotion_timestamp
            if sync_delay < 0.4:
                sync_color = (0, 255, 0)  # Verde ottimo
                delay_text = f"{sync_delay:.1f}s ✓"
            elif sync_delay < 0.8:
                sync_color = (255, 165, 0)  # Arancione accettabile
                delay_text = f"{sync_delay:.1f}s"
            else:
                sync_color = (255, 0, 0)  # Rosso problematico
                delay_text = f"{sync_delay:.1f}s!"
            
            # 1. Cornice Camera
            draw.rectangle([x, y, x+sw, y+sh], outline=cam_col_rgb, width=2)
            
            # 2. Tag Camera con indicatore batch/stabilità
            tag_w = 150
            tag_h = 28
            
            # Sfondo tag (nerro con trasparenza)
            draw.rectangle([x, y, x+tag_w, y+tag_h], fill=(0,0,0,200))
            
            # Accento colore + indicatore stabilità emozione
            if hasattr(frame_obj, 'emotion_streak') and frame_obj.emotion_streak > 3:
                stability_color = (0, 200, 0)  # Verde per stabile
                stab_text = " ✓"
            else:
                stability_color = cam_col_rgb
                stab_text = ""
            
            draw.rectangle([x, y, x+3, y+tag_h], fill=stability_color)
            
            # Nome camera + indicatore stabilità
            cam_text = f"{name}{stab_text}"
            draw.text((x + 10, y + 5), cam_text, font=font_cam_label, fill=(200, 200, 200))
            
            # Indicatore sincronizzazione (cerchio)
            sync_x = x + tag_w - 20
            sync_y = y + tag_h // 2
            draw.ellipse([sync_x-4, sync_y-4, sync_x+4, sync_y+4], fill=sync_color)
            
            # 3. Overlay emozione
            overlay = Image.new('RGBA', (sw, 60), (0,0,0,0))
            dr_ov = ImageDraw.Draw(overlay)
            dr_ov.rectangle([0, 0, sw, 60], fill=(0,0,0,180))
            dashboard.paste(overlay, (x, y + sh - 60), mask=overlay)
            
            # 4. Onda Sonora
            draw_waveform(draw, x + 15, y + sh - 45, 80, 30, emo_col_rgb)
            
            # 5. Emozione + timestamp
            emo_text = frame_obj.emotion.upper()
            bbox = draw.textbbox((0, 0), emo_text, font=font_emo)
            txt_w = bbox[2] - bbox[0]
            
            draw.text((x + sw - txt_w - 15, y + sh - 42), emo_text, font=font_emo, fill=emo_col_rgb)
            draw.text((x + sw - txt_w - 15, y + sh - 55), delay_text, font=font_tech, fill=sync_color)
        
        # 4. FOOTER con info batch
        footer_text = f"BATCH PROCESSING ACTIVE | Target: 4 FPS/stream | Last update: {datetime.datetime.now().strftime('%H:%M:%S')}"
        center_text(draw, footer_text, font_tech, total_w, total_h - 20, (80, 80, 80))
        
        # Conversione Finale
        final_np = np.array(dashboard)
        final_bgr = cv2.cvtColor(final_np, cv2.COLOR_RGB2BGR)
        
        with shared_data.lock:
            ret, buffer = cv2.imencode('.jpg', final_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ret:
                shared_data.output_frame = buffer.tobytes()
        
        time.sleep(0.033)  # ~30 FPS dashboard

# --- SERVER WEB (RIMANE COSÌ) ---
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
        time.sleep(0.033)  # Match dashboard FPS

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("--- MUSIC4D BATCH PROCESSING SERVER ---")
    print("Features: Batch inference, Dynamic rate limiting, Stability caching")
    print(f"Config: Batch size up to {len(SOURCES_CONFIG)}, Target: 4 FPS per stream")
    
    try:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map="auto", local_files_only=True 
        ).eval()
        processor = PaliGemmaProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        print("✅ AI Engine Ready for Batch Processing.")
    except Exception as e:
        print(f"❌ Errore AI: {e}")
        exit()
    
    # Ottimizzazione aggiuntiva: torch.compile se disponibile
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✅ Model compiled with torch.compile")
        except:
            print("⚠️ torch.compile not available")
    
    for conf in SOURCES_CONFIG:
        t = threading.Thread(target=video_worker_pipe, args=(conf,))
        t.daemon = True
        t.start()
        time.sleep(0.1)  # Stagger startup
    
    threading.Thread(target=analysis_worker, args=(model, processor), daemon=True).start()
    threading.Thread(target=compositor_worker, daemon=True).start()
    
    print(f"🚀 DASHBOARD: http://0.0.0.0:{WEB_PORT}")
    print("📊 Monitoraggio attivo - Guarda i log per le statistiche batch")
    
    app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True)