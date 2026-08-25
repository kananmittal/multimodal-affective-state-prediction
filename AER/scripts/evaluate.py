import torch
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.config import config
from src.models.affective_encoder import AffectiveEncoder
from src.data.dataset import AudioEmotionDataset
from tqdm import tqdm

EMOTION_MAP = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fear",
    5: "Surprise",
    6: "Disgust"
}

def evaluate(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_true = []
    val_error = 0
    arousal_error = 0
    total_samples = 0
    
    json_results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch["input_values"].to(device)
            cat_labels = batch["categorical_label"].to(device)
            valence_labels = batch["valence"].to(device)
            arousal_labels = batch["arousal"].to(device)
            file_paths = batch["file_path"] # Automatically aggregated into a tuple by PyTorch default_collate
            
            outputs = model(input_values)
            logits = outputs["categorical_logits"]
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            confidences = probs[torch.arange(probs.size(0)), preds]
            
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(cat_labels.cpu().numpy())
            
            val_error += torch.abs(outputs["valence"] - valence_labels).sum().item()
            arousal_error += torch.abs(outputs["arousal"] - arousal_labels).sum().item()
            total_samples += cat_labels.size(0)
            
            # Map predictions for JSON output
            pred_list = preds.cpu().numpy()
            true_list = cat_labels.cpu().numpy()
            conf_list = confidences.cpu().numpy()
            for i in range(len(file_paths)):
                json_results.append({
                    "file_path": file_paths[i],
                    "true_emotion": EMOTION_MAP.get(true_list[i], "Unknown"),
                    "predicted_emotion": EMOTION_MAP.get(pred_list[i], "Unknown"),
                    "emotion_confidence": round(float(conf_list[i]), 4)
                })
                
    # Core Scikit-Learn Metrics calculation
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_true)).float().mean().item()
    uar = recall_score(all_true, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_true, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
    
    mae_val = val_error / total_samples if total_samples > 0 else 0
    mae_ar = arousal_error / total_samples if total_samples > 0 else 0
    
    print("\n" + "="*40)
    print("🎯 EVALUATION METRICS REPORT")
    print("="*40)
    print(f"Categorical Accuracy : {accuracy * 100:.2f}%")
    print(f"UAR (Avg Recall)     : {uar * 100:.2f}%")
    print(f"Macro Precision      : {precision * 100:.2f}%")
    print(f"Macro F1-Score       : {f1 * 100:.2f}%")
    print("-" * 40)
    print(f"Valence MAE          : {mae_val:.4f}")
    print(f"Arousal MAE          : {mae_ar:.4f}")
    print("="*40 + "\n")
    
    # 1. Save JSON Report
    json_path = config.processed_dir / f"evaluation_results_{config.run_name}.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"✅ Saved individual JSON predictions to {json_path}")
    
    # 2. Generate Confusion Matrix Heatmap
    cm = confusion_matrix(all_true, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[EMOTION_MAP[i] for i in range(7)],
                yticklabels=[EMOTION_MAP[i] for i in range(7)])
    plt.ylabel('Actual Emotion (Ground Truth)', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.title(f'AER Confusion Matrix ({config.run_name.replace("_", " ").title()} Run)', fontsize=14, pad=15)
    plt.tight_layout()
    
    cm_path = config.processed_dir / f"confusion_matrix_{config.run_name}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"✅ Generated Confusion Matrix heatmap at {cm_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Advanced Evaluation Pipeline on {device}...")
    
    # 1. Load Model Architecture & Weights
    model = AffectiveEncoder(config)
    
    # Automatically find the latest checkpoint
    import glob
    import os
    checkpoint_pattern = str(config.base_dir / f"aer_{config.run_name}_epoch_*.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        print(f"Error: No trained weights found matching {checkpoint_pattern}")
        sys.exit(1)
        
    # Get the checkpoint with the highest epoch number
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_epoch_')[-1].split('.pt')[0]))
    model_weight_path = Path(latest_checkpoint)
    print(f"-> Auto-detected latest checkpoint: {model_weight_path.name}")
    
    if not model_weight_path.exists():
        print(f"Error: Could not find trained weights at {model_weight_path}")
        sys.exit(1)
        
    print(f"Loading weights from {model_weight_path}...")
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    
    # 2. Load the Blind Test Manifest with Versioning
    manifest_path = config.processed_dir / f"test_manifest_{config.run_name}.csv"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}. Please re-run preprocess_datasets.py first.")
        sys.exit(1)
        
    test_df = pd.read_csv(manifest_path)
    print(f"Successfully loaded {len(test_df)} unseen audio files for strict evaluation.")
    
    # 3. Initialize DataLoader
    test_dataset = AudioEmotionDataset(
        file_paths=test_df['file_path'].values,
        categorical_labels=test_df['label_idx'].values,
        valence_labels=test_df['valence'].values,
        arousal_labels=test_df['arousal'].values
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size * 2, # Can double batch size during eval since no gradients
        shuffle=False, 
        num_workers=4
    )
    
    # 5. Execute
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()
