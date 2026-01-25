"""
Data processing components for Instant Policy.

Key features:
- Sinusoidal positional encoding (NeRF-like)
- Edge features with relative position encoding
- Temporal edges within demonstrations
- Ghost gripper nodes for action representation
- Pseudo-demonstrations with procedural objects
- ShapeNet-based demonstrations with real 3D meshes
"""

from ip_src.data.graph_builder import (
    GraphBuilder,
    LocalGraph,
    ContextGraph,
    sinusoidal_positional_encoding,
    compute_edge_features,
)
from ip_src.data.pseudo_demo import (
    PseudoDemoGenerator,
    PseudoTaskConfig,
    MotionType,
    ObjectState,
    MotionPrimitive,
    resample_trajectory,
)
from ip_src.data.dataset import InstantPolicyDataset, collate_fn

# Optional: ShapeNet demo generator (requires trimesh)
try:
    from ip_src.data.shapenet_demo import (
        ShapeNetDemoGenerator,
        ShapeNetTaskConfig,
        ShapeNetLoader,
    )
    SHAPENET_AVAILABLE = True
except ImportError:
    SHAPENET_AVAILABLE = False
    ShapeNetDemoGenerator = None
    ShapeNetTaskConfig = None
    ShapeNetLoader = None

__all__ = [
    # Graph building
    "GraphBuilder",
    "LocalGraph",
    "ContextGraph",
    "sinusoidal_positional_encoding",
    "compute_edge_features",
    # Pseudo-demo generation
    "PseudoDemoGenerator",
    "PseudoTaskConfig",
    "MotionType",
    "ObjectState",
    "MotionPrimitive",
    "resample_trajectory",
    # Dataset
    "InstantPolicyDataset",
    "collate_fn",
    # ShapeNet (optional)
    "ShapeNetDemoGenerator",
    "ShapeNetTaskConfig",
    "ShapeNetLoader",
    "SHAPENET_AVAILABLE",
]
