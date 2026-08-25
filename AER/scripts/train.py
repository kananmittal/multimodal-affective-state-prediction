import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import sys
from pathlib import Path

# Fix python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.config import config
from src.models.affective_encoder import AffectiveEncoder
from src.data.dataset import AudioEmotionDataset

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    cat_criterion = nn.CrossEntropyLoss()
    dim_criterion = nn.MSELoss()
    
    # Progress bar wrapping the dataloader
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move to device
        input_values = batch["input_values"].to(device)
        cat_labels = batch["categorical_label"].to(device)
        valence_labels = batch["valence"].to(device)
        arousal_labels = batch["arousal"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_values)
        
        # Calculate losses
        loss_cat = cat_criterion(outputs["categorical_logits"], cat_labels)
        loss_val = dim_criterion(outputs["valence"], valence_labels)
        loss_ar = dim_criterion(outputs["arousal"], arousal_labels)
        
        # Combine losses (tune weights based on priority)
        loss = loss_cat + 0.5 * loss_val + 0.5 * loss_ar
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update UI trace
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)

def main():
    print("Initializing AER Training Pipeline...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    model = AffectiveEncoder(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 0. Checkpoint Resumption Logic
    import glob
    start_epoch = 0
    checkpoint_pattern = str(config.base_dir / f"aer_{config.run_name}_epoch_*.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if checkpoints:
        # Get the checkpoint with the highest epoch number
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_epoch_')[-1].split('.pt')[0]))
        latest_epoch = int(latest_checkpoint.split('_epoch_')[-1].split('.pt')[0])
        
        print(f"-> Found existing checkpoint: {Path(latest_checkpoint).name}")
        print(f"-> Resuming training from Epoch {latest_epoch + 1}")
        
        # Load weights into model
        model.load_state_dict(torch.load(latest_checkpoint, map_location=device))
        start_epoch = latest_epoch + 1
        
    if start_epoch >= config.epochs:
        print("Training already completed for the specified number of epochs!")
        sys.exit(0)
    
    # 1. Load Dataset Manifest with Versioning
    manifest_path = config.processed_dir / f"train_manifest_{config.run_name}.csv"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}. Did you run preprocess_datasets.py?")
        sys.exit(1)
        
    print(f"Loading data from: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    dataset = AudioEmotionDataset(
        file_paths=df['file_path'].values,
        categorical_labels=df['label_idx'].values,
        valence_labels=df['valence'].values,
        arousal_labels=df['arousal'].values
    )
    
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    print(f"Successfully loaded {len(dataset)} samples. Ready for Phase 1 Base Model Training.")
    
    # Example execution looping structure (usually you run this on Ubuntu to prevent Mac thermal issues)
    for epoch in range(start_epoch, config.epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{config.epochs} ---")
        epoch_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1} Completed. Avg Loss: {epoch_loss:.4f}")
        # Save model checkpoint with Versioning
        checkpoint_path = config.base_dir / f"aer_{config.run_name}_epoch_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"-> Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
