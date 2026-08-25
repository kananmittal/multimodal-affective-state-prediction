import torch
import torchaudio
import json
import sys
from pathlib import Path

# Setup paths
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config.config import config
from src.models.affective_encoder import AffectiveEncoder

EMOTION_MAP = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fear",
    5: "Surprise",
    6: "Disgust"
}

def load_and_preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != config.target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config.target_sample_rate)
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    waveform = waveform.squeeze(0)
    # Zero-Mean Unit-Variance required for Wav2Vec2 inference
    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
        
    return waveform.unsqueeze(0) # [1, Seq_Len]

def run_aer_inference(model, target_audio):
    """
    Modular inference function to be called by external routers (e.g. Flask).
    Returns a dictionary payload with the results.
    """
    input_values = load_and_preprocess_audio(target_audio)
    
    # Automatically move input tensor to the same device as the model
    device = next(model.parameters()).device
    input_values = input_values.to(device)
    
    with torch.no_grad():
        outputs = model(input_values)
        
    # Process outputs
    logits = outputs["categorical_logits"]
    probabilities = torch.softmax(logits, dim=1)
    
    pred_class_idx = torch.argmax(logits, dim=1).item()
    emotion_confidence = probabilities[0, pred_class_idx].item()
    emotion_label = EMOTION_MAP.get(pred_class_idx, "Unknown")
    
    valence_score = outputs["valence"].item()
    arousal_score = outputs["arousal"].item()
    
    latent_vector = outputs["latent_vector"].squeeze(0).tolist()
    
    response_payload = {
        "status": "success",
        "model": "AER_Encoder",
        "results": {
            "predicted_emotion": emotion_label,
            "emotion_confidence": round(emotion_confidence, 4),
            "valence": round(valence_score, 4),
            "arousal": round(arousal_score, 4),
            "latent_vector_dimension": len(latent_vector),
            "latent_vector_preview": latent_vector[:5] # Showing first 5 for brevity
        }
    }
    
    return response_payload

def main():
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_audio_file>")
        sys.exit(1)
        
    target_audio = sys.argv[1]
    
    device = torch.device("cpu") # For inference testing
    
    print("Loading AER Model Architecture...")
    model = AffectiveEncoder(config)
    
    # Locate and load the most freshly trained checkpoint
    model_weight_path = Path(__file__).resolve().parent.parent / "aer_cultural_mixed_epoch_9.pt"
    if model_weight_path.exists():
        print(f"Loading trained weights from {model_weight_path}...")
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
    else:
        print(f"WARNING: Weights {model_weight_path} not found. Running with untrained base model!")
        
    model.eval()
    
    print(f"Ingesting audio: {target_audio}")
    print("Encoding...")
    
    response_payload = run_aer_inference(model, target_audio)
    
    print("\n--- INFERENCE RESULTS ---")
    print(json.dumps(response_payload, indent=4))
    print("-------------------------\n")
    print("Latent Vector successfully extracted. Ready for Vector Database ingestion!")

if __name__ == "__main__":
    main()
