"""
Model components for Instant Policy.

Implements the graph-based flow matching architecture from the paper:
- σ(·): LocalGraphEncoder - encodes individual timestep observations
- φ(·): ContextAggregator - aggregates demonstrations with current observation  
- ψ(·): ActionDecoder - Graph Transformer that outputs geometric flow vectors

Key features:
- Edge-aware attention (Equation 3)
- Ghost gripper nodes for action representation
- Geometric flow output (not flat noise)
"""

from ip_src.models.graph_diffusion import GraphDiffusion
from ip_src.models.encoders import (
    PointNetPlusPlusEncoder,
    SAModule,
    sinusoidal_positional_encoding,
    load_encoder_from_checkpoint,
)
from ip_src.models.graph_transformer import (
    GraphTransformer,
    GraphTransformerLayer,
    EdgeAwareAttention,
    HeteroEdgeAwareAttention,
    LocalGraphEncoder,
    ContextAggregator,
    ActionDecoder,
)
from ip_src.models.diffusion import (
    FlowMatchingScheduler,
    FlowMatchingConfig,
    PositionToTransform,
    compute_flow_loss,
)

__all__ = [
    # Main model
    "GraphDiffusion",
    # Encoders (matching official pretrained weights)
    "PointNetPlusPlusEncoder",
    "SAModule",
    "sinusoidal_positional_encoding",
    "load_encoder_from_checkpoint",
    # Graph Transformers
    "GraphTransformer",
    "GraphTransformerLayer",
    "EdgeAwareAttention",
    "HeteroEdgeAwareAttention",
    "LocalGraphEncoder",
    "ContextAggregator",
    "ActionDecoder",
    # Flow Matching
    "FlowMatchingScheduler",
    "FlowMatchingConfig",
    # Utilities
    "PositionToTransform",
    "compute_flow_loss",
]
