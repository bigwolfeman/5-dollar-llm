"""
TitanMAC configuration for 5-dollar-llm training pipeline.

Provides configuration classes that match the interface of MoEModelConfig
but configure TitanMAC (neural memory + sliding window attention).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TitanMACModelConfig:
    """
    Configuration for TitanMAC model.

    Matches the interface of MoEModelConfig for compatibility with
    the training infrastructure, while configuring TitanMAC-specific
    features (neural memory, sliding window attention).

    Architecture Overview:
        - Token + position embeddings
        - Stack of TitanBlock layers (windowed attention + MLP)
        - Optional neural long-term memory (Titans paper feature)
        - Output RMSNorm + LM head

    Key Differences from MoE:
        - No mixture of experts (single FFN per layer)
        - Neural memory provides auxiliary loss instead of load balancing
        - Uses sliding window attention instead of full attention
    """

    # Model architecture
    d_model: int = 1536
    n_heads: int = 12
    n_layers: int = 26
    d_ff: int = 4096

    # Training parameters (matching MoEModelConfig interface)
    batch_size: int = 8
    max_steps: int = 10000
    gradient_accumulation_steps: int = 12
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    adamw_lr: float = 0.003
    warmup_ratio: float = 0.05

    # Data parameters
    max_seq_len: int = 512
    num_documents: int = 2000
    max_tokens: int = 500000

    # Evaluation
    eval_every: int = 10
    eval_steps: int = 100

    # Regularization
    weight_decay: float = 0.2
    dropout: float = 0.1
    grad_clip: float = 1.0

    # Technical
    # NOTE: AMP is disabled by default for TitanMAC because neural memory's
    # gradient-based updates (torch.autograd.grad) require consistent precision.
    # The memory MLP parameters are float32 but AMP casts activations to float16,
    # which causes gradient computation failures.
    use_amp: bool = False
    vocab_size: Optional[int] = None
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    # =========================================================================
    # TitanMAC-specific parameters
    # =========================================================================

    # Attention configuration
    window_size: int = 512  # Sliding window attention size
    n_persistent: int = 16  # Number of persistent (global) tokens
    use_block_sparse: bool = True  # Use O(T*w) block-sparse attention
    block_size: int = 64  # Block size for block-sparse attention

    # Embeddings
    tie_weights: bool = True  # Tie input/output embeddings

    # =========================================================================
    # Neural Memory (Titans paper, arxiv 2501.00663)
    # =========================================================================
    use_neural_memory: bool = True  # Enable gradient-based neural memory
    n_memory_layers: int = 2  # Number of layers in memory MLP (paper: L_M >= 2)
    d_memory: Optional[int] = None  # Memory dimension (defaults to d_model)
    memory_theta_lr: float = 0.01  # Learning rate for memory updates
    memory_forget_hidden: int = 32  # Hidden dim for forget gate MLP
    memory_decay_hidden: int = 32  # Hidden dim for decay gate MLP

    # Architecture variant: "MAC", "MAG", or "MAL"
    # MAC: Memory as Context (concatenated dataflow)
    # MAG: Memory as Gate (multiplicative gating) - recommended for efficiency
    # MAL: Memory as Layer (replaces attention in some layers)
    titans_variant: str = "MAG"

    # MAC-specific parameters
    segment_size: int = 512  # Segment size for MAC processing
    n_memory_tokens: int = 32  # Memory tokens retrieved per segment

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        # Set d_memory default
        if self.d_memory is None:
            self.d_memory = self.d_model


@dataclass
class TitanMACGPU24GBConfig(TitanMACModelConfig):
    """
    TitanMAC configuration optimized for 24GB GPUs (RTX 4090, etc).

    Designed to be comparable to GPU24GBMoEModelConfig:
        - Similar parameter count (~50M total)
        - Same effective batch size (via gradient accumulation)
        - Same learning rate schedule

    Key Features:
        - Neural memory enabled (MAG variant for efficiency)
        - Sliding window attention (512 tokens)
        - Persistent tokens for global context

    NOTE: TitanMAC requires AMP disabled due to neural memory gradient updates.
    To compensate for increased memory usage, we use:
        - Smaller batch size (4 instead of 16)
        - Gradient accumulation (4x to maintain effective batch size)
        - Gradient checkpointing enabled by default
    """

    # Reduced architecture for 24GB VRAM
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048

    # Batch size: Reduced from 16 to 4, with 4x gradient accumulation
    # This maintains effective batch size of 16 while fitting in memory
    # without AMP (which is disabled for neural memory compatibility)
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Training parameters (optimized via sweep, same as MoE)
    muon_lr: float = 0.04
    adamw_lr: float = 0.006

    # Data (matching MoE baseline)
    max_seq_len: int = 1024
    num_documents: int = 25000

    # Logging and training length
    log_milestones: Tuple[int, ...] = (100, 200, 300)
    max_steps: int = 800
    eval_every: int = 50

    # TitanMAC-specific settings for 24GB
    window_size: int = 512  # Sliding window size
    n_persistent: int = 8  # Reduced persistent tokens
    use_block_sparse: bool = True
    block_size: int = 64

    # Neural memory settings
    use_neural_memory: bool = True
    n_memory_layers: int = 2
    d_memory: Optional[int] = None  # Will default to d_model (512)
    memory_theta_lr: float = 0.01
    memory_forget_hidden: int = 32
    memory_decay_hidden: int = 32

    # Use MAG variant (most memory efficient)
    titans_variant: str = "MAG"
    segment_size: int = 512
    n_memory_tokens: int = 16  # Reduced for memory efficiency

    # Regularization (matching baseline)
    dropout: float = 0.1
    weight_decay: float = 0.2
    grad_clip: float = 1.0

    # Enable gradient checkpointing by default to save memory
    use_gradient_checkpointing: bool = True

    def __post_init__(self):
        super().__post_init__()


@dataclass
class DebugTitanMACConfig(TitanMACModelConfig):
    """
    Tiny TitanMAC configuration for fast debugging on any hardware.

    Matches DebugMoEConfig interface for quick testing.
    """

    # Tiny architecture
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512

    # Batch size
    batch_size: int = 2
    gradient_accumulation_steps: int = 1

    # Training parameters
    muon_lr: float = 0.01
    adamw_lr: float = 0.001

    # Data
    max_seq_len: int = 128
    num_documents: int = 100

    # Reduced logging
    log_milestones: Tuple[int, ...] = (10, 50, 80)
    max_steps: int = 100
    eval_every: int = 10

    # TitanMAC-specific settings for debug
    window_size: int = 64
    n_persistent: int = 4
    use_block_sparse: bool = True
    block_size: int = 32

    # Neural memory (simplified for debug)
    use_neural_memory: bool = True
    n_memory_layers: int = 2
    d_memory: Optional[int] = None
    memory_theta_lr: float = 0.01
    memory_forget_hidden: int = 16
    memory_decay_hidden: int = 16

    # Use MAG variant (simplest)
    titans_variant: str = "MAG"
    segment_size: int = 64
    n_memory_tokens: int = 8

    def __post_init__(self):
        super().__post_init__()
