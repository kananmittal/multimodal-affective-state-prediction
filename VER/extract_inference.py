import cv2
import csv
import datetime
import os
import glob
import sys

# ================= CONFIGURAZIONE =================
BASE_DIR = "session_data"
VIDEO_DIR = os.path.join(BASE_DIR, "full_recordings")
OUTPUT_DIR = os.path.join(BASE_DIR, "final_dataset")
IMAGES_SUBDIR = os.path.join(OUTPUT_DIR, "images")

# Quale sorgente vuoi estrarre? (DIRECTOR, ORCHESTRA, AUDIENCE)
TARGET_SOURCE = "ORCHESTRA"
# ==================================================

def parse_csv_timestamp(ts_str):
    """Timestamp dal log (precisione microsecondi)"""
    return datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S_%f")

def parse_video_timestamp(filename):
    """Timestamp dal nome file video (precisione secondi)"""
    try:
        base = os.path.basename(filename)
        parts = os.path.splitext(base)[0].split("_")
        # Gestisce nomi tipo REC_DIRECTOR_20231030_153010.mp4
        ts_str = f"{parts[-2]}_{parts[-1]}"
        return datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except Exception as e:
        return None

def find_video_file(source, video_dir):
    """Trova il video più recente per la source specificata"""
    pattern = os.path.join(video_dir, f"REC_{source}_*.mp4")
    files = glob.glob(pattern)
    return sorted(files)[-1] if files else None

def find_latest_log_csv(base_dir):
    """Trova il file di log più recente (inference_log_*.csv)"""
    pattern = os.path.join(base_dir, "inference_log_*.csv")
    files = glob.glob(pattern)
    return sorted(files)[-1] if files else None

def main():
    # 1. Creazione Struttura Cartelle
    if not os.path.exists(IMAGES_SUBDIR):
        os.makedirs(IMAGES_SUBDIR)

    # 2. Trova il Log CSV più recente
    input_log_csv = find_latest_log_csv(BASE_DIR)
    if not input_log_csv:
        print(f"❌ Nessun file di log trovato in {BASE_DIR}")
        return
    print(f"📂 Log trovato: {os.path.basename(input_log_csv)}")

    # 3. Trova il Video
    video_path = find_video_file(TARGET_SOURCE, VIDEO_DIR)
    if not video_path:
        print(f"❌ Video non trovato per {TARGET_SOURCE}")
        return

    print(f"🎬 Video trovato: {os.path.basename(video_path)}")
    
    video_start_time = parse_video_timestamp(video_path)
    if not video_start_time:
        print("❌ Errore data video.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Errore apertura file video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 4. Generazione Nome Output CSV (Target + Data Corrente)
    extraction_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_csv_name = f"dataset_labels_{TARGET_SOURCE}_{extraction_ts}.csv"
    output_dataset_csv_path = os.path.join(OUTPUT_DIR, output_csv_name)
    
    print(f"📄 Creazione Dataset CSV: {output_csv_name}")
    
    processed_count = 0

    with open(input_log_csv, 'r') as f_in, open(output_dataset_csv_path, 'w', newline='') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # Header del CSV Dataset Finale
        writer.writerow(["filename", "label", "source", "timestamp"])
        
        header = next(reader, None) # Salta header input

        print("🚀 Estrazione frame RAW in corso...")

        for row in reader:
            if not row: continue
            
            # Parsing Input: [Timestamp, Source, Emotion, Inf_Time]
            ts_str, source, emotion = row[0], row[1], row[2]

            if source != TARGET_SOURCE:
                continue

            try:
                # Sincronizzazione Temporale
                event_time = parse_csv_timestamp(ts_str)
                delta = event_time - video_start_time
                seconds_offset = delta.total_seconds()

                if seconds_offset < 0: continue

                target_frame = int(seconds_offset * fps)

                if target_frame >= total_frames:
                    break

                # Posizionamento sul frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()

                if ret:
                    # Nome file immagine
                    filename = f"img_{source}_{ts_str}.jpg"
                    filepath = os.path.join(IMAGES_SUBDIR, filename)
                    
                    # 1. Salvataggio Immagine RAW
                    cv2.imwrite(filepath, frame)
                    
                    # 2. Scrittura nel CSV Output
                    writer.writerow([filename, emotion, source, ts_str])
                    
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        sys.stdout.write(f"\r📸 Processati: {processed_count}")
                        sys.stdout.flush()

            except Exception as e:
                # Debug solo se serve
                # print(f"Skip: {e}")
                continue

    cap.release()
    print(f"\n\n✅ OPERAZIONE COMPLETATA!")
    print(f"📂 Frame salvati in:  {IMAGES_SUBDIR}")
    print(f"📄 Dataset Label CSV: {output_dataset_csv_path}")
    print(f"📊 Totale Frame:      {processed_count}")

if __name__ == "__main__":
    main()