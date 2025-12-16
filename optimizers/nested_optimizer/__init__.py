"""
Nested Optimizer package for MoE models.

Implements the Nested Learning paradigm from NeurIPS 2025:
- DeepNestedOptimizer: Main optimizer with learned momentum and LR control
- NestedController: Learned LR multipliers per parameter group
- L2RegressionMomentum: Neural network momentum with L2 regression loss
- Parameter grouping utilities for MoE models

Reference: "Nested Learning: The Illusion of Deep Learning Architectures"
           Behrouz et al., NeurIPS 2025
"""

from .deep_nested_optimizer import (
    DeepNestedOptimizer,
    L2RegressionMomentum,
    ContinuumMemoryState,
)
from .nested_controller import NestedController
from .param_groups import group_moe_params, group_titanmac_params, infer_param_depth
from .meta_trainer import (
    UnrolledMetaTrainer,
    SimplifiedMetaTrainer,
    create_meta_trainer,
)

__all__ = [
    'DeepNestedOptimizer',
    'L2RegressionMomentum',
    'ContinuumMemoryState',
    'NestedController',
    'group_moe_params',
    'group_titanmac_params',
    'infer_param_depth',
    'UnrolledMetaTrainer',
    'SimplifiedMetaTrainer',
    'create_meta_trainer',
]
