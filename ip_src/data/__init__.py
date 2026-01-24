"""
Data processing components for Instant Policy.

Key features:
- Sinusoidal positional encoding (NeRF-like)
- Edge features with relative position encoding
- Temporal edges within demonstrations
- Ghost gripper nodes for action representation
"""

from ip_src.data.graph_builder import (
    GraphBuilder,
    LocalGraph,
    ContextGraph,
    sinusoidal_positional_encoding,
    compute_edge_features,
)
from ip_src.data.pseudo_demo import PseudoDemoGenerator
from ip_src.data.dataset import InstantPolicyDataset, collate_fn

__all__ = [
    "GraphBuilder",
    "LocalGraph",
    "ContextGraph",
    "sinusoidal_positional_encoding",
    "compute_edge_features",
    "PseudoDemoGenerator",
    "InstantPolicyDataset",
    "collate_fn",
]
