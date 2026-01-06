"""
Meta-Tool Configuration
=======================
Central configuration for all hyperparameters and settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """Configuration for base and hypernetwork models.
    
    GPU MEMORY PRESETS - Choose based on your available VRAM:
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │ GPU VRAM    │ Base Model                      │ Encoder             │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 24GB+       │ meta-llama/Llama-3.2-3B-Instruct│ Llama-3.2-1B        │
    │ (4090/A6000)│ base_model_dim=3072, layers=28  │ encoder_dim=2048    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 12-16GB     │ microsoft/phi-2                 │ all-MiniLM-L6-v2    │
    │ (3090/4080) │ base_model_dim=2560, layers=32  │ encoder_dim=384     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 8-10GB      │ TinyLlama/TinyLlama-1.1B-Chat   │ all-MiniLM-L6-v2    │
    │ (3080/3070) │ base_model_dim=2048, layers=22  │ encoder_dim=384     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 4-6GB/CPU   │ facebook/opt-350m               │ all-MiniLM-L6-v2    │
    │ (Testing)   │ base_model_dim=512, layers=24   │ encoder_dim=384     │
    └─────────────────────────────────────────────────────────────────────┘
    """
    # DEFAULT: Medium config (~12-16GB VRAM for training, ~8GB inference)
    base_model_name: str = "microsoft/phi-2"
    base_model_dim: int = 2560  # Phi-2 hidden dimension
    base_model_layers: int = 32
    
    # Documentation encoder (small, efficient - only 80MB)
    encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    encoder_dim: int = 384  # MiniLM output dimension
    
    # LoRA configuration
    lora_rank: int = 16  # r
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj"])
    
    # Hypernetwork
    latent_dim: int = 1024  # d_latent (smaller for efficiency)
    attention_dim: int = 512  # d_attn
    num_adapt_layers: int = 32  # |L_adapt| - should match base_model_layers
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32


@dataclass
class TrainingConfig:
    """Configuration for meta-training."""
    # Meta-learning
    num_episodes: int = 50000
    batch_size: int = 2  # Tools per batch (reduce if OOM)
    support_set_size: int = 5  # K (examples per tool)
    query_set_size: int = 3  # N_query
    
    # Gradient accumulation (effective_batch = batch_size * accumulation_steps)
    gradient_accumulation_steps: int = 4
    
    # Optimization
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Value function
    value_lr: float = 1e-4
    td_gamma: float = 0.99
    value_hidden_dims: List[int] = field(default_factory=lambda: [1024, 512])
    
    # Checkpointing
    save_every: int = 1000
    eval_every: int = 500
    checkpoint_dir: str = "./checkpoints_xlarge"


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    # Beam search
    beam_width: int = 5  # B
    max_depth: int = 10  # D
    candidates_per_step: int = 3  # K_cand
    
    # Generation
    max_new_tokens: int = 64  # Reduced for faster eval - outputs are short (SQL, commands, API calls)
    temperature: float = 1.0  # 1.0 = greedy decoding (more stable)
    top_p: float = 0.9
    do_sample: bool = False  # Greedy by default - faster and more consistent
    
    # Memory
    episodic_k: int = 3  # k_epi
    schema_k: int = 5  # k_schema
    confidence_threshold: float = 0.7  # τ_conf


@dataclass
class DataConfig:
    """Configuration for data loading."""
    # Paths
    toolbench_path: str = "./data/toolbench"
    gorilla_path: str = "./data/gorilla"
    spider_path: str = "./data/spider2"
    webarena_path: str = "./data/webarena"
    intercode_path: str = "./data/intercode"
    
    # Processing
    max_doc_tokens: int = 4096
    max_trajectory_tokens: int = 2048
    
    # Meta-training tools
    num_meta_train_tools: int = 500


@dataclass
class MetaToolConfig:
    """Master configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment
    experiment_name: str = "meta_tool_v1"
    seed: int = 42
    wandb_project: Optional[str] = "meta-tool"
    

def get_config(model_size: str = "medium") -> MetaToolConfig:
    """Get configuration with model size preset.
    
    Args:
        model_size: One of "tiny", "small", "medium", "large", "xlarge"
            - tiny: ~4GB VRAM (OPT-350M) - for testing/CPU
            - small: ~8GB VRAM (TinyLlama-1.1B) - RTX 3070/3080
            - medium: ~12GB VRAM (Phi-2) - RTX 3090/4080 [DEFAULT]
            - large: ~16GB VRAM (Llama-3.2-3B + MiniLM encoder) - RTX 4090/A6000
            - xlarge: ~32GB VRAM (Llama-3.2-3B + Llama-1B encoder) - A100/H100
    """
    config = MetaToolConfig()
    
    if model_size == "tiny":
        # ~4GB VRAM - for testing or CPU
        config.model.base_model_name = "facebook/opt-350m"
        config.model.base_model_dim = 512
        config.model.base_model_layers = 24
        config.model.encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.model.encoder_dim = 384
        config.model.latent_dim = 256
        config.model.attention_dim = 256
        config.model.num_adapt_layers = 24
        config.training.batch_size = 4
        
    elif model_size == "small":
        # ~8GB VRAM - RTX 3070/3080
        config.model.base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        config.model.base_model_dim = 2048
        config.model.base_model_layers = 22
        config.model.encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.model.encoder_dim = 384
        config.model.latent_dim = 512
        config.model.attention_dim = 512
        config.model.num_adapt_layers = 22
        config.training.batch_size = 2
        
    elif model_size == "medium":
        # ~12GB VRAM - RTX 3090/4080 (DEFAULT - already set)
        pass
        
    elif model_size == "large":
        # ~12-16GB VRAM with 4-bit quantization - RTX 4090/A6000/A100
        config.model.base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
        config.model.base_model_dim = 3072
        config.model.base_model_layers = 28
        # Use efficient encoder to save memory for the main model
        config.model.encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.model.encoder_dim = 384
        config.model.latent_dim = 512
        config.model.attention_dim = 512
        config.model.num_adapt_layers = 28
        config.training.batch_size = 1  # Small batch with gradient accumulation
        config.training.gradient_accumulation_steps = 8
        
    elif model_size == "xlarge":
        # ~24-32GB VRAM - Uses Llama for BOTH encoder and generator
        # For A100/H100 or multi-GPU setups
        config.model.base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
        config.model.base_model_dim = 3072
        config.model.base_model_layers = 28
        # Use Llama-1B as encoder (same family, better alignment)
        config.model.encoder_model_name = "meta-llama/Llama-3.2-1B-Instruct"
        config.model.encoder_dim = 2048
        config.model.latent_dim = 1024
        config.model.attention_dim = 1024
        config.model.num_adapt_layers = 28
        config.training.batch_size = 1
        config.training.gradient_accumulation_steps = 16
        
    else:
        raise ValueError(f"Unknown model_size: {model_size}. Choose from: tiny, small, medium, large, xlarge")
    
    return config
