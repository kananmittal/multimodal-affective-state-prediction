import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AffectiveEncoder(nn.Module):
    """
    Core Architecture for the AER model.
    Maps raw audio to a spatial latent vector and derives 
    both categorical and dimensional (Valence/Arousal) emotional parameters.
    """
    def __init__(self, config):
        super(AffectiveEncoder, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.model_name, use_safetensors=True)
        
        if config.freeze_feature_extractor:
            # Updated to prevent Transformers v5 deprecation error
            if hasattr(self.wav2vec2, "freeze_feature_encoder"):
                self.wav2vec2.freeze_feature_encoder()
            else:
                self.wav2vec2.freeze_feature_extractor()
                
        # Since Wav2Vec2 lacks standard token embeddings, enable_input_require_grads() crashes.
        # Checkpointing is not structurally necessary on your Ubuntu GPU for this base model.
        if hasattr(self.wav2vec2, "gradient_checkpointing_disable"):
            self.wav2vec2.gradient_checkpointing_disable()
            
        hidden_size = self.wav2vec2.config.hidden_size # 768 for base models
        
        # Multi-task Classification Heads
        self.categorical_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.num_categorical_labels)
        )
        
        self.dimensional_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2) # [Valence, Arousal] coordinates
        )
        
    def forward(self, input_values, attention_mask=None):
        # Extract audio features up to the transformer layers.
        # output_hidden_states=True forces it to explicitly expose the vector layout.
        outputs = self.wav2vec2(
            input_values=input_values, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # The latent vector is resolved here.
        # Mean pooling across the temporal dimension creates the single 768-D representation
        # which can then be fed flawlessly into a Vector Database.
        last_hidden_state = outputs.hidden_states[-1]
        latent_vector = last_hidden_state.mean(dim=1)
        
        # Inference heads
        categorical_logits = self.categorical_classifier(latent_vector)
        dimensional_outputs = self.dimensional_classifier(latent_vector)
        
        valence = dimensional_outputs[:, 0]
        arousal = dimensional_outputs[:, 1]
        
        return {
            "categorical_logits": categorical_logits,
            "valence": valence,
            "arousal": arousal,
            "latent_vector": latent_vector # Crucial output payload
        }
