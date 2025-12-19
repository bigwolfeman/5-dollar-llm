"""
HOPE (Hybrid Optimized Persistent Encoder) configuration.

HOPE combines delta-rule memory (Titans L2) with standard attention
for efficient long-context modeling.

Reference: Titans paper - https://arxiv.org/abs/2501.00663
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HOPEModelConfig:
    """
    Base configuration for HOPE model.

    HOPE architecture:
        - Token + position embeddings
        - Stack of HOPE blocks (delta-rule memory + attention + FFN)
        - Output LayerNorm + LM head

    Key features:
        - Delta-rule memory: M = M - α(M @ k) @ k.T + β @ v @ k.T
        - Chunked attention for memory efficiency
        - Learnable α (forget) and β (write) parameters
    """

    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048

    # Sequence parameters
    max_seq_len: int = 1024
    chunk_size: int = 64  # Chunk size for delta-rule memory processing

    # Data
    num_documents: int = 250000

    # Training parameters
    batch_size: int = 16
    max_steps: int = 10000
    gradient_accumulation_steps: int = 1
    adamw_lr: float = 3e-4
    warmup_ratio: float = 0.05

    # Evaluation
    eval_every: int = 50
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.01
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    use_amp: bool = False  # Delta-rule memory benefits from full precision
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # HOPE-specific parameters
    memory_scale: float = 0.1  # Scale for memory contribution to output
    alpha_init: float = 0.0  # Initial logit for forget gate (sigmoid -> 0.5)
    beta_init: float = 0.0  # Initial logit for write gate (sigmoid -> 0.5)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


@dataclass
class HOPEGPU24GBConfig(HOPEModelConfig):
    """
    HOPE configuration for 24GB GPUs.

    Scaled to ~45M parameters (similar to HOPE-nano reference).
    """

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048

    batch_size: int = 16
    gradient_accumulation_steps: int = 1

    adamw_lr: float = 3e-4

    max_seq_len: int = 1024
    chunk_size: int = 64

    log_milestones: Tuple[int, ...] = (100, 200, 300)
    max_steps: int = 800
    eval_every: int = 50

    def __post_init__(self):
        super().__post_init__()


@dataclass
class HOPE168MConfig(HOPEModelConfig):
    """
    HOPE configuration scaled to ~168M parameters.

    Matches MoE baseline parameter count for fair architecture comparison.

    Parameter breakdown:
        - Token embeddings: vocab_size * d_model = 50257 * 1024 = 51.5M
        - Per layer: ~12.6M (attention/memory + FFN)
        - 9 layers: ~113M
        - Total: ~165M (approximately matches MoE 168M)

    NOTE: HOPE's delta-rule memory is sequential within chunks,
    which makes it significantly slower than attention-only models.
    This is a fundamental architecture tradeoff, not a bug.
    """

    # Scaled architecture to ~168M params
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 9
    d_ff: int = 4096

    # Batch size: batch=8 * grad_accum=2 = effective 16
    batch_size: int = 8
    gradient_accumulation_steps: int = 2

    # Training parameters
    adamw_lr: float = 1e-4  # Lower LR for larger model

    # Data
    max_seq_len: int = 1024
    num_documents: int = 250000  # Match MoE/TitanMAC
    chunk_size: int = 64

    # Logging
    log_milestones: Tuple[int, ...] = (100, 200, 300)
    max_steps: int = 800
    eval_every: int = 50

    # Regularization
    dropout: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # HOPE-specific
    memory_scale: float = 0.1

    def __post_init__(self):
        super().__post_init__()


@dataclass
class DebugHOPEConfig(HOPEModelConfig):
    """
    Tiny HOPE configuration for fast debugging on any hardware.
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512

    batch_size: int = 2
    gradient_accumulation_steps: int = 1

    adamw_lr: float = 1e-3

    max_seq_len: int = 128
    num_documents: int = 100
    chunk_size: int = 32

    log_milestones: Tuple[int, ...] = (10, 50, 80)
    max_steps: int = 100
    eval_every: int = 10

    def __post_init__(self):
        super().__post_init__()
