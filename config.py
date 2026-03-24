"""
Configuration for the small LLM project.
Optimized for laptop (8GB RAM, CPU-only) while remaining configurable.
All model and training parameters are centralized here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union


# ---------------------------------------------------------------------------
# Model architecture (easy to tune for size vs quality)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    GPT-style transformer architecture.
    Tune these for your hardware: smaller = faster, less memory; larger = better quality.
    """

    # Vocabulary (set automatically from tokenizer; default only for init)
    vocab_size: int = 50257

    # Context window: max tokens the model can see at once (affects memory heavily)
    max_seq_length: int = 256  # 128 is CPU-friendly; use 256+ if you have more RAM/GPU

    # Embedding size: dimension of each token representation
    d_model: int = 384  # 128 for 8GB RAM; 256 or 512 for larger machines

    # Number of attention heads (must divide d_model evenly)
    n_heads: int = 8  # 4 or 8 typical for small models

    # Number of transformer layers (stack depth)
    n_layers: int = 10  # 4–6 for laptop; 6–12 for larger training

    # Feed-forward hidden size (often 4 * d_model)
    d_ff: int = 1536  # 4 * d_model when d_model=128

    dropout: float = 0.1

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


# ---------------------------------------------------------------------------
# Training (CPU-friendly defaults)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training loop: learning rate scheduling, gradient clipping, checkpointing."""

    # Data: single path or list of paths (comma-separated string or list)
    data_path: Union[str, List[str]] = "data/train.txt"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 500
    log_every: int = 50

    # Optimization (small batch + grad accumulation = effective large batch on CPU)
    batch_size: int = 8  # Small for 8GB RAM
    gradient_accumulation_steps: int = 4  # Effective batch = batch_size * this
    learning_rate: float = 2e-4
    max_steps: int = 40_000
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Learning rate schedule: "linear_warmup" or "cosine"
    lr_schedule: str = "cosine"  # Cosine decay after warmup is often better

    # Device
    device: str = "cuda"  # Prefer "cpu" for 8GB laptop; use "cuda" if GPU available

    # Tokenizer
    tokenizer_path: Optional[str] = None
    vocab_size: int = 4096  # BPE vocab size when building from data


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass
class GenerateConfig:
    """Text generation: temperature, top-k, top-p (nucleus) sampling."""

    max_new_tokens: int = 100
    temperature: float = 0.8  # Higher = more random
    top_k: int = 40  # 0 = disabled
    top_p: float = 1.0  # 1.0 = disabled
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Feedback / learning from user
# ---------------------------------------------------------------------------

FEEDBACK_FILE = "feedback.jsonl"
DEFAULT_CHECKPOINT = "checkpoints/checkpoint_final.pt"
DEFAULT_TOKENIZER_DIR = "checkpoints/tokenizer"

# Default config instances
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_GENERATE_CONFIG = GenerateConfig()
