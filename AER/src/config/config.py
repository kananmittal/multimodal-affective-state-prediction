import dataclasses
from pathlib import Path

@dataclasses.dataclass
class AERConfig:
    # Pipeline Versioning
    run_name: str = "cultural_mixed" # Switch to "cultural_mixed" to activate the new datasets
    
    # Model parameters
    model_name: str = "facebook/wav2vec2-base"
    num_categorical_labels: int = 7 # 0: Neutral, 1: Happy, 2: Sad, 3: Angry, 4: Fear, 5: Surprise, 6: Disgust
    
    # Audio parameters
    target_sample_rate: int = 16000
    max_duration_seconds: int = 5
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 2e-5 # Scaled down; 1e-4 causes catastrophic forgetting in pre-trained transformers
    epochs: int = 10
    freeze_feature_extractor: bool = True
    
    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    raw_generic_dir: Path = data_dir / "raw" / "generic"
    raw_cultural_dir: Path = data_dir / "raw" / "cultural"
    processed_dir: Path = data_dir / "processed"
    
    def __post_init__(self):
        # Ensure directories exist
        self.raw_generic_dir.mkdir(parents=True, exist_ok=True)
        self.raw_cultural_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

config = AERConfig()
