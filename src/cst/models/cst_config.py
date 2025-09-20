"""
Contextual Spectrum Tokenization (CST) Configuration
Core configuration classes for the CST implementation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import yaml


@dataclass
class CSTConfig:
    """Main configuration class for CST model"""
    
    # Model Architecture
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    vocab_size: int = 30522
    max_sequence_length: int = 512
    dropout: float = 0.1
    
    # CST Module Configuration
    char_vocab_size: int = 256
    char_embed_dim: int = 64
    context_window: int = 64
    fragment_encoding_dim: int = 256
    document_encoding_dim: int = 128
    metadata_dim: int = 64
    multimodal_dim: int = 512
    hidden_dim: int = 512
    fused_dim: int = 384
    embed_dim: int = 64
    
    # External Dimensions
    raw_doc_dim: int = 1024
    clip_dim: int = 512
    audio_dim: int = 768
    context_feature_dim: int = 128
    
    # Ambiguity Classification
    ambiguous_word_ids: List[int] = field(default_factory=list)
    ambiguity_threshold: float = 0.5
    
    # Caching
    cache_size: int = 10000
    l1_cache_size: int = 5000
    
    # Training Configuration
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    warmup_steps: int = 10000
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Contrastive Learning
    temperature: float = 0.07
    contrastive_weight: float = 0.3
    mlm_weight: float = 0.7
    
    # Regularization
    reference_update_freq: int = 1000
    reference_momentum: float = 0.999
    drift_regularization_weight: float = 0.1
    
    # Production Settings
    max_batch_size: int = 64
    inference_cache_ttl: int = 3600
    
    # Metadata Processing
    num_authors: int = 10000
    num_domains: int = 100
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'CSTConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def update(self, **kwargs) -> 'CSTConfig':
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        return self


@dataclass
class TrainingConfig:
    """Training-specific configuration"""
    
    # Data paths
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    test_data_path: str = "data/test"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_steps: int = 5000
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 1000
    wandb_project: Optional[str] = None
    tensorboard_dir: str = "tensorboard_logs"
    
    # Distributed Training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"
    
    # Mixed Precision
    use_amp: bool = True
    amp_opt_level: str = "O1"


@dataclass
class ProductionConfig:
    """Production deployment configuration"""
    
    # Serving
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    health_check_interval: int = 60
    
    # Caching
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8080
    enable_tracing: bool = True
    trace_sampling_rate: float = 0.1
    
    # Resource Limits
    max_memory_usage: float = 0.8  # Fraction of available GPU memory
    max_cpu_usage: float = 0.8     # Fraction of available CPU cores
    
    # Model Serving
    model_path: str = "models/cst_model.pt"
    device: str = "cuda"
    precision: str = "fp16"  # fp32, fp16, int8


# Default configurations
DEFAULT_CONFIG = CSTConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_PRODUCTION_CONFIG = ProductionConfig()
