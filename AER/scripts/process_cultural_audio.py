import os
import sys
import pandas as pd
from pathlib import Path
import subprocess
from pydub import AudioSegment
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.config import config

# Links provided by the user
YOUTUBE_LINKS = {
    "Kota factory episode 1.csv": "https://youtu.be/JWbnEt3xuos?si=VXJxWXYshWdOqxsj",
    "kota factory episode 2.csv": "https://youtu.be/zW-fcAGzEew?si=5WpIXtqNly28WTT7",
    "kota factory episode 3.csv": "https://youtu.be/gIwgSpEg6ZY?si=66PzzH3wL-HXTMUu",
    "kota factory episode 4.csv": "https://youtu.be/SaCgKXQiXGE?si=gcB4JfXJkZJzGV7X",
    "kota factory episode 5.csv": "https://youtu.be/923ks1pc0LQ?si=163P0LRFrsGvflQY"
}

# Emotion Mapping to convert complex CSV labels into the strictly defined 7-class taxonomy
EMOTION_MAPPING = {
    "Neutral": "Neutral",
    "Joy": "Happy",
    "Sadness": "Sad",
    "Anger": "Angry",
    "Fear": "Fear",
    "Surprise": "Surprise",
    "Disgust": "Disgust",
    
    # Complex mappings (Resolving mixed emotions to primary dominance based on acoustic urgency)
    "Neutral+Anger": "Angry",
    "Sadness+Anger": "Angry",
    "not purely Positive": "Neutral",
    "Anger+Neutral": "Angry",
    "Anger++Neutral": "Angry",
    "Joy+Surprise": "Happy",
    "Surprise+Joy": "Happy",
    "Sadness+Neutral": "Sad",
    "Fear+Anger": "Fear",
    "Disgust+Anger": "Disgust"
}

def map_emotion(raw_label):
    if pd.isna(raw_label):
        return "Neutral"
    
    label = str(raw_label).strip()
    
    if label in EMOTION_MAPPING:
        return EMOTION_MAPPING[label]
        
    # Default fallback
    return "Neutral"

def download_audio_from_yt(url, output_path):
    print(f"\n=> Triggering yt-dlp engine for {url}")
    # Extract audio as best quality wav. Requires ffmpeg to be installed on the system!
    base_path = str(output_path).replace(".wav", "")
    
    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--output", f"{base_path}.%(ext)s",
        url
    ]
    
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("\n❌ CRITICAL ERROR: 'yt-dlp' or 'ffmpeg' is not installed on the system.")
        print("Please run: sudo apt-get install ffmpeg")
        sys.exit(1)

def process_episode(csv_path, yt_url, cultural_dir, output_dir):
    csv_name = csv_path.name
    episode_id = csv_name.replace(".csv", "").replace(" ", "_").lower()
    
    full_audio_path = cultural_dir / f"{episode_id}_full.wav"
    
    if not full_audio_path.exists():
        download_audio_from_yt(yt_url, full_audio_path)
        
    print(f"\n=> Loading full episode into RAM: {full_audio_path}")
    try:
        episode_audio = AudioSegment.from_wav(str(full_audio_path))
    except Exception as e:
        print(f"Failed to load audio for {episode_id}: {e}")
        return
        
    df = pd.read_csv(csv_path)
    
    print(f"=> Slicing {len(df)} discrete utterances based on CSV timestamps...")
    saved_count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=episode_id):
        start_sec = row.get('start_seconds', 0)
        end_sec = row.get('end_seconds', 0)
        
        # Validate timestamps
        if pd.isna(start_sec) or pd.isna(end_sec) or end_sec <= start_sec:
            continue
            
        start_ms = int(float(start_sec) * 1000)
        end_ms = int(float(end_sec) * 1000)
        
        raw_emotion = row.get('Emotion presented', 'Neutral')
        mapped_emotion = map_emotion(raw_emotion)
        
        # Perform memory slice
        utterance_audio = episode_audio[start_ms:end_ms]
        
        # Save out to discrete file for Neural Network ingestion
        filename = f"{episode_id}_utt_{idx}_{mapped_emotion}.wav"
        save_path = output_dir / filename
        
        utterance_audio.export(str(save_path), format="wav")
        saved_count += 1
        
    print(f"✅ Extracted {saved_count} audio files for {episode_id}.")
        
def main():
    print("==================================================")
    print("🚀 INITIALIZING CULTURAL DATASET PROCESSOR (KOTA)")
    print("==================================================")
    
    cultural_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "cultural"
    output_dir = cultural_dir / "processed_audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for csv_file, url in YOUTUBE_LINKS.items():
        csv_path = cultural_dir / csv_file
        if csv_path.exists():
            process_episode(csv_path, url, cultural_dir, output_dir)
        else:
            print(f"⚠️ Warning: Expected CSV '{csv_file}' not found in {cultural_dir}. Skipping.")
            
    print("\n✅ DONE! All cultural data successfully ripped, sliced, and standardized.")
    print(f"Audio chunks are waiting in: {output_dir}")

if __name__ == "__main__":
    main()
