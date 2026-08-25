import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

# Ensure config loads
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.config import config

# The 7-Class Unified Taxonomy
# 0: Neutral, 1: Happy, 2: Sad, 3: Angry, 4: Fear, 5: Surprise, 6: Disgust
UNIFIED_EMOTION_MAP = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fear': 4,
    'surprise': 5,
    'disgust': 6
}

# Optional: Valence / Arousal mapping approximations per class (Scale: -1.0 to 1.0)
V_A_APPROXIMATION = {
    0: (0.0, 0.0),    # Neutral
    1: (0.8, 0.6),    # Happy (Positive, High Arousal)
    2: (-0.8, -0.6),  # Sad (Negative, Low Arousal)
    3: (-0.8, 0.8),   # Angry (Negative, High Arousal)
    4: (-0.6, 0.9),   # Fear (Negative, Very High Arousal)
    5: (0.4, 0.8),    # Surprise (Positive/Neutral, High Arousal)
    6: (-0.9, 0.4)    # Disgust (Very Negative, Moderate Arousal)
}

def parse_ravdess(base_dir):
    records = []
    # Format: 03-01-05-01-01-01-01.wav -> Emotion is the 3rd token (05=angry)
    # RAVDESS map: 01=neutral, 02=calm(map->neutral), 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprise
    rav_map = {'01':'neutral', '02':'neutral', '03':'happy', '04':'sad', '05':'angry', '06':'fear', '07':'disgust', '08':'surprise'}
    
    # Use case-insensitive rglob to support Ext4/Ubuntu filesystems
    files = [str(f) for f in base_dir.rglob("*.wav") if "ravdess" in str(f).lower()]
    for f in files:
        basename = os.path.basename(f).split('.')[0]
        parts = basename.split('-')
        if len(parts) >= 3:
            emo_code = parts[2]
            if emo_code in rav_map:
                records.append({
                    'file_path': f,
                    'dataset': 'RAVDESS',
                    'emotion': rav_map[emo_code],
                    'label_idx': UNIFIED_EMOTION_MAP[rav_map[emo_code]]
                })
    return records

def parse_tess(base_dir):
    records = []
    seen_files = set()
    # Format: folder determines emotion (e.g. OAF_angry, YAF_happy)
    # Suffix is also _angry, _happy, _ps (pleasant surprise), _fear, _sad, _disgust, _neutral
    # Use case-insensitive rglob to support Ext4/Ubuntu filesystems
    files = [str(f) for f in base_dir.rglob("*.wav") if "tess" in str(f).lower() or "toronto" in str(f).lower()]
    for f in files:
        basename = os.path.basename(f).lower()
        if basename in seen_files:
            continue
        seen_files.add(basename)
        if 'neutral' in basename: emo = 'neutral'
        elif 'happy' in basename: emo = 'happy'
        elif 'sad' in basename: emo = 'sad'
        elif 'angry' in basename: emo = 'angry'
        elif 'fear' in basename: emo = 'fear'
        elif 'disgust' in basename: emo = 'disgust'
        elif 'ps' in basename or 'surprise' in basename: emo = 'surprise'
        else: continue
            
        records.append({
            'file_path': f,
            'dataset': 'TESS',
            'emotion': emo,
            'label_idx': UNIFIED_EMOTION_MAP[emo]
        })
    return records

def parse_crema(base_dir):
    records = []
    # Format: 1001_DFA_ANG_XX.wav -> 3rd token is emotion
    # CREMA map: NEU, HAP, SAD, ANG, FEA, DIS
    crema_map = {'NEU':'neutral', 'HAP':'happy', 'SAD':'sad', 'ANG':'angry', 'FEA':'fear', 'DIS':'disgust'}
    
    # Use case-insensitive rglob to support Ext4/Ubuntu filesystems
    files = [str(f) for f in base_dir.rglob("*.wav") if "crema" in str(f).lower()]
    for f in files:
        basename = os.path.basename(f).split('.')[0]
        parts = basename.split('_')
        if len(parts) >= 3:
            emo_code = parts[2]
            if emo_code in crema_map:
                records.append({
                    'file_path': f,
                    'dataset': 'CREMA',
                    'emotion': crema_map[emo_code],
                    'label_idx': UNIFIED_EMOTION_MAP[crema_map[emo_code]]
                })
    return records

def parse_savee(base_dir):
    records = []
    # Format: DC_a01.wav or JE_sa01.wav -> first letters before digits is emotion
    # SAVEE map: a=angry, d=disgust, f=fear, h=happy, n=neutral, sa=sad, su=surprise
    savee_map = {'a':'angry', 'd':'disgust', 'f':'fear', 'h':'happy', 'n':'neutral', 'sa':'sad', 'su':'surprise'}
    
    # Use case-insensitive rglob to support Ext4/Ubuntu filesystems
    files = [str(f) for f in base_dir.rglob("*.wav") if "savee" in str(f).lower()]
    for f in files:
        basename = os.path.basename(f).split('_')[-1] # sa01.wav -> sa01.wav
        
        # strip numbers and .wav
        import re
        emo_code = re.sub(r'[0-9]+', '', basename.split('.')[0]).lower()
        
        if emo_code in savee_map:
            records.append({
                'file_path': f,
                'dataset': 'SAVEE',
                'emotion': savee_map[emo_code],
                'label_idx': UNIFIED_EMOTION_MAP[savee_map[emo_code]]
            })
    return records

def parse_iemocap(base_dir):
    import re
    from pathlib import Path
    records = []
    
    # OS-agnostic search for EmoEvaluation bypassing exact root naming
    eval_files = glob.glob(str(base_dir / "**" / "EmoEvaluation" / "*.txt"), recursive=True)
    
    # IEMOCAP direct to class mapping (Drops 'fru' and 'xxx' automatically)
    iemocap_map = {'neu': 'neutral', 'hap': 'happy', 'exc': 'happy', 'sad': 'sad', 'ang': 'angry', 'fea': 'fear', 'sur': 'surprise', 'dis': 'disgust'}
    
    pattern = re.compile(r'^\[.+\]\s+(Ses\w+)\s+(\w+)\s+\[(\d+\.\d+),\s+(\d+\.\d+),\s+(\d+\.\d+)\]')
    
    for eval_file_str in eval_files:
        # Ignore MacOS hidden artifact files if they accidentally copied over
        if "/._" in eval_file_str or "__MACOSX" in eval_file_str:
            continue
            
        with open(eval_file_str, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = pattern.match(line.strip())
                if match:
                    file_id = match.group(1)
                    emo_code = match.group(2)
                    val_score = float(match.group(3))
                    act_score = float(match.group(4))
                    
                    if emo_code in iemocap_map:
                        emotion = iemocap_map[emo_code]
                        session = file_id[4] # E.g., '1' from Ses01
                        
                        dialog_prefix = "_".join(file_id.split('_')[:-1])
                        
                        # Agnostic Path Leaping: Go up 3 levels to Session directory, then dive into wavs
                        session_dir = Path(eval_file_str).parent.parent.parent
                        wav_path = session_dir / "sentences" / "wav" / dialog_prefix / f"{file_id}.wav"
                        
                        if wav_path.exists():
                            norm_val = (val_score - 3.0) / 2.0
                            norm_ar = (act_score - 3.0) / 2.0
                            
                            records.append({
                                'file_path': str(wav_path),
                                'dataset': 'IEMOCAP',
                                'emotion': emotion,
                                'label_idx': UNIFIED_EMOTION_MAP[emotion],
                                'valence': round(norm_val, 4),
                                'arousal': round(norm_ar, 4)
                            })
    return records

def parse_cultural(base_dir):
    records = []
    # Cultural audio is saved in data/raw/cultural/processed_audio
    cultural_dir = base_dir.parent / "cultural" / "processed_audio"
    if not cultural_dir.exists():
        return records
        
    # Files are named like: kota_factory_episode_1_utt_5_Happy.wav
    files = [str(f) for f in cultural_dir.rglob("*.wav")]
    for f in files:
        basename = os.path.basename(f)
        # The mapped emotion is always the last token before .wav
        parts = basename.replace(".wav", "").split("_")
        emo_str = parts[-1].lower()
        
        if emo_str in UNIFIED_EMOTION_MAP:
            records.append({
                'file_path': f,
                'dataset': 'KOTA_FACTORY',
                'emotion': emo_str,
                'label_idx': UNIFIED_EMOTION_MAP[emo_str]
            })
    return records

def generate_distribution_chart(df, save_path):
    plt.figure(figsize=(10, 6))
    
    # Custom elegant color palette ensuring aesthetics
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x='emotion', hue='dataset', data=df, palette='viridis')
    
    plt.title('Audio Samples per Emotion Class across Datasets (System Config: 7-Class)', fontsize=14, pad=15)
    plt.xlabel('Emotion Classification', fontsize=12)
    plt.ylabel('Number of Audio Samples', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Sub-Dataset Origin', loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"-> Saved distribution chart to {save_path}")

def main():
    print("Initiating Dataset Standardizer (M1 AER Pipeline)...")
    
    base_raw_dir = config.raw_generic_dir
    if not base_raw_dir.exists():
        print(f"Error: Directory {base_raw_dir} does not exist.")
        return

    # Aggregate records
    all_records = []
    all_records.extend(parse_ravdess(base_raw_dir))
    all_records.extend(parse_tess(base_raw_dir))
    all_records.extend(parse_crema(base_raw_dir))
    all_records.extend(parse_savee(base_raw_dir))
    all_records.extend(parse_iemocap(base_raw_dir))
    
    # Selective Loading: Only pull cultural data if the run_name specifies it
    if config.run_name == "cultural_mixed":
        print("-> Pulling Kota Factory Cultural Data...")
        all_records.extend(parse_cultural(base_raw_dir))
    
    df = pd.DataFrame(all_records)
    
    if len(df) == 0:
        print("No audio files found! Check directory structures inside data/raw/generic/")
        return
        
    print(f"Successfully aggregated {len(df)} total audio samples across {df['dataset'].nunique()} datasets.")
    
    # Enrich with approximated Valence/Arousal values ONLY for generic datasets missing them
    df['valence'] = df.apply(lambda row: V_A_APPROXIMATION[row['label_idx']][0] if pd.isna(row.get('valence')) else row['valence'], axis=1)
    df['arousal'] = df.apply(lambda row: V_A_APPROXIMATION[row['label_idx']][1] if pd.isna(row.get('arousal')) else row['arousal'], axis=1)
    
    # Stratified 90/10 Split to prevent data leakage and guarantee valid evaluation
    # Using stratify=df['emotion'] to ensure both Train and Test sets have the same proportion of emotion classes
    train_df, test_df = train_test_split(df, test_size=0.10, random_state=42, stratify=df['emotion'])
    
    print(f"Data Split Complete: {len(train_df)} Training Samples, {len(test_df)} Testing Samples.")
    
    # Save Split Manifests with Versioning
    train_manifest_path = config.processed_dir / f"train_manifest_{config.run_name}.csv"
    test_manifest_path = config.processed_dir / f"test_manifest_{config.run_name}.csv"
    
    train_df.to_csv(train_manifest_path, index=False)
    test_df.to_csv(test_manifest_path, index=False)
    print(f"-> Saved universal training manifest to {train_manifest_path}")
    print(f"-> Saved blind testing manifest to {test_manifest_path}")
    
    # Generate Chart (using full dataset for holistic visualization)
    chart_path = config.processed_dir / f"class_distribution_{config.run_name}.png"
    generate_distribution_chart(df, chart_path)
    
    print("\nClass Breakdown summary:")
    print(df['emotion'].value_counts())
    print("\nDataset fully synchronized and standardized to the 7-Class Unified Taxonomy.")

if __name__ == "__main__":
    main()
