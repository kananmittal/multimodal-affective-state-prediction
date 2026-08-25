import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

# Fix python parsing context by importing sys path if needed, but relative import assumes running as a module.
try:
    from src.config.config import config
except ImportError:
    from config.config import config

class AudioEmotionDataset(Dataset):
    def __init__(self, file_paths, categorical_labels, valence_labels, arousal_labels):
        self.file_paths = file_paths
        self.categorical_labels = categorical_labels
        self.valence_labels = valence_labels
        self.arousal_labels = arousal_labels
        
        self.target_sample_rate = config.target_sample_rate
        self.max_length = config.target_sample_rate * config.max_duration_seconds
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        path = str(self.file_paths[idx])
        waveform, sample_rate = torchaudio.load(path)
        
        # Enforce 16kHz resample
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Truncate or pad
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            padding_length = self.max_length - waveform.shape[1]
            if padding_length > 0:
                waveform = torch.nn.functional.pad(waveform, (0, padding_length))
                
        waveform = waveform.squeeze(0)
        
        # Wav2Vec2 natively requires Zero-Mean Unit-Variance scaling, otherwise 
        # the internal LayerNorms clip the gradients to zero (causing loss freezing)
        waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var() + 1e-7)
        
        return {
            "input_values": waveform,
            "categorical_label": torch.tensor(self.categorical_labels[idx], dtype=torch.long),
            "valence": torch.tensor(self.valence_labels[idx], dtype=torch.float),
            "arousal": torch.tensor(self.arousal_labels[idx], dtype=torch.float),
            "file_path": path
        }
