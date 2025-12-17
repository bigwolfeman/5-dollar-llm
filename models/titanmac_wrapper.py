"""
TitanMAC Wrapper for 5-dollar-llm training pipeline.

Wraps the TitanMAC model to match the MoEMinimalLLM interface:
- forward(x, return_aux_loss=True) returns (logits, aux_loss) tuple
- aux_loss is the memory_loss from TitanMAC output
"""

import sys
import os
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

# Add TitanMAC to path
_titan_path = os.path.join(os.path.dirname(__file__), "..", "111TitanMAC-Standalone")
if _titan_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_titan_path))

from titans_core.models.titanmac import TitanMAC
from titans_core.config import TitanMACConfig


class TitanMACWrapper(nn.Module):
    """
    Wrapper around TitanMAC to match MoEMinimalLLM interface.

    The MoE training loop expects:
        logits, aux_loss = model(x, return_aux_loss=True)

    TitanMAC returns:
        {"logits": ..., "memory_loss": ..., "loss": ...}

    This wrapper bridges the two interfaces.

    Args:
        config: TitanMACConfig or compatible config object

    Example:
        >>> from configs.titanmac_config import TitanMACGPU24GBConfig
        >>> config = TitanMACGPU24GBConfig()
        >>> model = TitanMACWrapper(config)
        >>> x = torch.randint(0, config.vocab_size, (2, 128))
        >>> logits, aux_loss = model(x, return_aux_loss=True)
    """

    def __init__(self, config):
        super().__init__()

        # Convert config to TitanMACConfig if needed
        if isinstance(config, TitanMACConfig):
            self.titan_config = config
        else:
            # Build TitanMACConfig from the wrapper config
            self.titan_config = TitanMACConfig(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                window_size=getattr(config, 'window_size', 512),
                n_persistent=getattr(config, 'n_persistent', 16),
                use_block_sparse=getattr(config, 'use_block_sparse', True),
                block_size=getattr(config, 'block_size', 64),
                dropout=getattr(config, 'dropout', 0.0),
                tie_weights=getattr(config, 'tie_weights', True),
                use_neural_memory=getattr(config, 'use_neural_memory', True),
                n_memory_layers=getattr(config, 'n_memory_layers', 2),
                d_memory=getattr(config, 'd_memory', None),
                memory_theta_lr=getattr(config, 'memory_theta_lr', 0.01),
                memory_forget_hidden=getattr(config, 'memory_forget_hidden', 32),
                memory_decay_hidden=getattr(config, 'memory_decay_hidden', 32),
                titans_variant=getattr(config, 'titans_variant', 'MAG'),
                segment_size=getattr(config, 'segment_size', 512),
                n_memory_tokens=getattr(config, 'n_memory_tokens', 32),
            )

        # Store original config for compatibility
        self.config = config

        # Create the TitanMAC model
        self.model = TitanMAC(self.titan_config)

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass matching MoEMinimalLLM interface.

        Args:
            x: Input token IDs [B, T]
            return_aux_loss: If True, return (logits, aux_loss) tuple

        Returns:
            If return_aux_loss=True: (logits, aux_loss) tuple
            If return_aux_loss=False: logits only

        Shape:
            Input: x [B, T] (long)
            Output: logits [B, T, vocab_size], aux_loss scalar (or None)
        """
        # TitanMAC expects (input_ids, labels, attention_mask)
        # We don't pass labels here - loss is computed in the training loop
        output = self.model(input_ids=x, labels=None, attention_mask=None)

        logits = output["logits"]

        if return_aux_loss:
            # Use memory_loss as aux_loss (analogous to MoE load balancing loss)
            aux_loss = output.get("memory_loss", None)

            # Ensure aux_loss is a scalar tensor, not None
            if aux_loss is None:
                aux_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

            return logits, aux_loss

        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters (for compatibility)."""
        return self.model.get_num_params(non_embedding=non_embedding)

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.model.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.model.disable_gradient_checkpointing()

    def reset_memory(self):
        """Reset memory state (for evaluation/inference)."""
        self.model.reset_memory()
        if hasattr(self.model, 'neural_memory') and self.model.neural_memory is not None:
            # Reset neural memory state if it exists
            pass  # Neural memory doesn't have persistent state to reset

    def get_neural_memory_stats(self):
        """
        Get neural memory statistics for saturation monitoring.

        Returns dict with:
            - memory_param_norm: L2 norm of memory MLP weights
            - momentum_norm: L2 norm of momentum buffer
            - alpha_t: Forget gate output (0=retain, 1=forget) - saturation indicator
            - eta_t: Decay gate output (momentum decay)
            - grad_norm: Memory gradient norm before clipping
            - grad_clipped: Whether gradient was clipped

        Saturation interpretation:
            - High alpha_t (>0.8) + high loss = saturated, constantly forgetting
            - Low alpha_t (<0.2) + low loss = healthy learning

        Returns:
            Dictionary with memory stats, or None if neural memory disabled
        """
        return self.model.get_neural_memory_stats()


def create_titanmac_model(config) -> TitanMACWrapper:
    """
    Factory function to create a TitanMAC model with wrapper.

    Args:
        config: Configuration object with model parameters

    Returns:
        TitanMACWrapper instance
    """
    model = TitanMACWrapper(config)

    # Initialize weights (TitanMAC does this internally)

    return model
