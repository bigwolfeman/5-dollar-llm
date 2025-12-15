"""
Parameter grouping utilities for DeepNestedOptimizer with MoE models.

Separates model parameters into:
- Core params: attention projections, expert FFN weights (2D matrices)
- Embed params: embeddings, norms, router weights (1D or special params)

This grouping aligns with Muon vs AdamW separation in the baseline,
enabling fair A/B comparison between optimizers.
"""

import re
from typing import Tuple, List
import torch.nn as nn


def group_moe_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Group MoE model parameters into core and embed groups.

    This grouping is designed to match the Muon+AdamW split from the baseline:
    - Core: 2D weight matrices (attention, experts) - these benefit from Muon/nested LR
    - Embed: embeddings, norms, router - these typically need AdamW-style treatment

    Args:
        model: MoEMinimalLLM model instance

    Returns:
        Tuple of (core_params, embed_params) lists

    Core params include:
        - Attention projections (qkv, w_o, query, compressed_kv, decompressed_kv)
        - Expert FFN weights (linear1, linear2)

    Embed params include:
        - Token embeddings
        - RMSNorm weights
        - Router gate weights
        - Any 1D parameters

    Example:
        >>> model = MoEMinimalLLM(config)
        >>> core, embed = group_moe_params(model)
        >>> print(f"Core: {len(core)}, Embed: {len(embed)}")
    """
    core_params = []
    embed_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Classification based on parameter name and shape
        is_embed_param = False

        # Token embeddings always go to embed group
        if 'token_embedding' in name:
            is_embed_param = True

        # Norms (RMSNorm, LayerNorm, etc.) go to embed group
        elif 'norm' in name.lower():
            is_embed_param = True

        # Router/gate weights go to embed group
        elif 'router' in name or 'gate' in name:
            is_embed_param = True

        # 1D parameters (biases, etc.) go to embed group
        elif param.ndim != 2:
            is_embed_param = True

        # Everything else (2D matrices) goes to core
        # This includes:
        # - attention.qkv, attention.w_o
        # - attention.query, attention.compressed_kv, attention.decompressed_kv
        # - experts.linear1, experts.linear2

        if is_embed_param:
            embed_params.append(param)
        else:
            core_params.append(param)

    return core_params, embed_params


def infer_param_depth(name: str, n_layers: int) -> float:
    """
    Infer relative depth of parameter from name.

    Args:
        name: Parameter name (e.g., "transformer_blocks.5.attention.qkv.weight")
        n_layers: Total number of layers

    Returns:
        Normalized depth in [0, 1] where 0=embedding, 1=output

    Example:
        >>> infer_param_depth("transformer_blocks.8.attention.qkv.weight", n_layers=16)
        0.5625  # Layer 8 is slightly past middle of 16-layer model

    Depth mapping:
        - Embeddings: 0.0
        - Layer i: (i + 1) / n_layers
        - Final norm: 1.0
        - LM head: 1.0
    """
    # Embedding layers: depth 0
    if 'embed' in name.lower() or 'token_embedding' in name:
        return 0.0

    # Output layers: depth 1
    if 'lm_head' in name.lower():
        return 1.0

    # Final norm before output
    if name.startswith('norm.') or name == 'norm.weight':
        return 1.0

    # Try to extract layer index from name
    # Patterns for MoE model: "transformer_blocks.5."
    layer_patterns = [
        r"transformer_blocks\.(\d+)\.",
        r"layers\.(\d+)\.",
        r"blocks\.(\d+)\.",
        r"layer\.(\d+)\.",
    ]

    for pattern in layer_patterns:
        match = re.search(pattern, name)
        if match:
            layer_idx = int(match.group(1))
            # Normalize to [0, 1] range
            # Layer 0 -> small positive depth
            # Layer n_layers-1 -> depth near 1
            depth = (layer_idx + 1) / n_layers
            return min(depth, 0.99)  # Cap at 0.99 to distinguish from output

    # Default: middle depth for unknown params
    return 0.5


def get_param_info(model: nn.Module) -> dict:
    """
    Get detailed information about parameter grouping.

    Useful for debugging and understanding the parameter split.

    Args:
        model: MoE model instance

    Returns:
        Dict with parameter counts and names by group
    """
    core_params, embed_params = group_moe_params(model)

    core_names = []
    embed_names = []

    for name, param in model.named_parameters():
        if param in core_params:
            core_names.append(name)
        elif param in embed_params:
            embed_names.append(name)

    return {
        'core_count': len(core_params),
        'embed_count': len(embed_params),
        'core_numel': sum(p.numel() for p in core_params),
        'embed_numel': sum(p.numel() for p in embed_params),
        'core_names': core_names,
        'embed_names': embed_names,
    }
