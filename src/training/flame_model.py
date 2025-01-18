# File: src/training/flame_model.py

import mlx.core as mx
import mlx.nn as nn
from .config import TrainingConfig

class FlameModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        TrainingConfig.log_step("Creating model architecture...")
        
        # Validate configuration
        self._validate_config()
        
        # Model dimensions
        self.hidden_size = config.model_config["hidden_size"]
        self.vocab_size = config.model_config["vocab_size"]
        self.num_heads = config.model_config["num_attention_heads"]
        self.num_layers = config.model_config["num_hidden_layers"]
        
        # Initialize components
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.transformer = nn.TransformerEncoder(
            self.num_layers,
            self.hidden_size,
            self.num_heads,
            4*self.hidden_size,
            0.1,
            nn.SiLU()
        )
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)
        
        TrainingConfig.log_step(f"✓ Model created with vocab size: {self.vocab_size}")
    
    def _validate_config(self):
        """Validate model configuration"""
        required_keys = [
            "hidden_size", "vocab_size", "num_attention_heads",
            "num_hidden_layers", "intermediate_size"
        ]
        
        for key in required_keys:
            if key not in self.config.model_config:
                raise ValueError(f"Missing required configuration: {key}")
        
        if self.config.model_config["vocab_size"] < 1000:
            raise ValueError(f"Suspiciously small vocab size: {self.config.model_config['vocab_size']}")
        
    def __call__(self, input_ids, attention_mask=None):
        if mx.metal.is_available():
            mx.metal.clear_cache()
        
        x = self.embedding(input_ids)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
        x = self.transformer(x, mask=attention_mask)
        
        if mx.metal.is_available():
            mx.metal.clear_cache()
            
        return self.lm_head(x)

    def load_weights(self, model_path):
        TrainingConfig.log_step("Loading model weights...")
        weights = mx.load(str(model_path / "model.safetensors"), format="safetensors")
        self.update(weights)
        TrainingConfig.log_step("✓ Model weights loaded successfully")