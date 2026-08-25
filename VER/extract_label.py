import cv2
import csv
import datetime
import os
import glob
import sys

# ================= CONFIGURAZIONE =================
# Cartella base
BASE_DIR = "session_data"
VIDEO_DIR = os.path.join(BASE_DIR, "full_recordings")

# Cartella di Output specifica per questa richiesta
OUTPUT_DIR = os.path.join(BASE_DIR, "joy_frames")

# L'emozione che stiamo cercando
TARGET_LABEL = "joy"

# Quale camera vuoi analizzare? (DIRECTOR, ORCHESTRA, AUDIENCE)
TARGET_SOURCE = "DIRECTOR"
# ==================================================

def parse_csv_timestamp(ts_str):
    return datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S_%f")

def parse_video_timestamp(filename):
    try:
        base = os.path.basename(filename)
        parts = os.path.splitext(base)[0].split("_")
        ts_str = f"{parts[-2]}_{parts[-1]}"
        return datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except Exception as e:
        return None

def find_video_file(source, video_dir):
    pattern = os.path.join(video_dir, f"REC_{source}_*.mp4")
    files = glob.glob(pattern)
    return sorted(files)[-1] if files else None

def find_latest_log_csv(base_dir):
    pattern = os.path.join(base_dir, "inference_log_*.csv")
    files = glob.glob(pattern)
    return sorted(files)[-1] if files else None

def main():
    # 1. Crea la cartella se non esiste
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Creata nuova cartella: {OUTPUT_DIR}")
    else:
        print(f"📂 Cartella output esistente: {OUTPUT_DIR}")

    # 2. Trova File
    csv_path = find_latest_log_csv(BASE_DIR)
    video_path = find_video_file(TARGET_SOURCE, VIDEO_DIR)

    if not csv_path or not video_path:
        print("❌ Errore: File CSV o Video non trovati.")
        return

    print(f"📄 Leggo Log: {os.path.basename(csv_path)}")
    print(f"🎬 Leggo Video: {os.path.basename(video_path)}")

    # 3. Setup Video
    video_start = parse_video_timestamp(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not video_start or not cap.isOpened():
        print("❌ Errore apertura video.")
        return

    count_joy = 0
    
    print(f"\n🚀 Cerco '{TARGET_LABEL}' nel video {TARGET_SOURCE}...")

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # Salta header

        for row in reader:
            if not row: continue
            
            # Parsing: Timestamp, Source, Emotion
            ts_str, source, emotion = row[0], row[1], row[2]

            # FILTRO 1: Deve essere la Source giusta
            if source != TARGET_SOURCE: continue

            # FILTRO 2: Deve essere "joy"
            if emotion.lower() != TARGET_LABEL: continue

            # Calcolo Tempo
            try:
                event_time = parse_csv_timestamp(ts_str)
                delta = event_time - video_start
                seconds = delta.total_seconds()
                
                if seconds < 0: continue
                
                # Calcolo Frame
                frame_id = int(seconds * fps)
                
                if frame_id >= total_frames: break

                # Estrazione
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()

                if ret:
                    # Salva Raw Frame
                    # Nome: joy_DIRECTOR_2023...jpg
                    out_name = f"{TARGET_LABEL}_{source}_{ts_str}.jpg"
                    out_path = os.path.join(OUTPUT_DIR, out_name)
                    
                    cv2.imwrite(out_path, frame)
                    count_joy += 1
                    
                    sys.stdout.write(f"\r📸 Frame 'JOY' salvati: {count_joy}")
                    sys.stdout.flush()

            except Exception as e:
                continue

    cap.release()
    print(f"\n\n✅ OPERAZIONE COMPLETATA!")
    print(f"📍 I frame sono qui: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()