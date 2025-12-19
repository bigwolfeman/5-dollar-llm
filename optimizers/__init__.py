"""Optimizers for 5-dollar-llm.

This module contains:
- Muon: Momentum-based optimizer with Newton-Schulz orthogonalization

Note: DeepNestedOptimizer and related nested learning components
have been moved to 111TitanMAC-Standalone/titans_core/opt/
"""

from .muon import Muon, zeropower_via_newtonschulz5

__all__ = [
    'Muon',
    'zeropower_via_newtonschulz5',
]
