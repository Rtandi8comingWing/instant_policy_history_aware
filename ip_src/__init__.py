"""
Instant Policy - In-Context Imitation Learning via Graph Diffusion

This module provides a pure Python implementation of the Instant Policy model,
compatible with the original instant_policy.so interface.
"""

from ip_src.models.graph_diffusion import GraphDiffusion
from ip_src.utils.sampling import sample_to_cond_demo

__all__ = [
    "GraphDiffusion",
    "sample_to_cond_demo",
]

__version__ = "1.0.0"
