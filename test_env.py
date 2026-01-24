"""
Environment verification script for Instant Policy.
Run this to check if all dependencies are correctly installed.
"""

import sys
print(f"Python: {sys.version}")

# Core dependencies
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import torch_geometric
    print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
except ImportError as e:
    print(f"✗ PyTorch Geometric: {e}")

try:
    from torch_geometric.nn import fps, knn
    print("✓ torch-cluster (fps, knn): OK")
except ImportError as e:
    print(f"✗ torch-cluster: {e}")

try:
    import torch_scatter
    print("✓ torch-scatter: OK")
except ImportError as e:
    print(f"✗ torch-scatter: {e}")

try:
    import pytorch_lightning as pl
    print(f"✓ PyTorch Lightning: {pl.__version__}")
except ImportError as e:
    print(f"✗ PyTorch Lightning: {e}")

try:
    import open3d as o3d
    print(f"✓ Open3D: {o3d.__version__}")
except ImportError as e:
    print(f"✗ Open3D: {e}")

try:
    import scipy
    print(f"✓ SciPy: {scipy.__version__}")
except ImportError as e:
    print(f"✗ SciPy: {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")

# Test project import
print("\n--- Testing ip_src imports ---")
try:
    from ip_src.models import GraphDiffusion, PointNetPlusPlusEncoder
    print("✓ ip_src.models: OK")
except ImportError as e:
    print(f"✗ ip_src.models: {e}")

try:
    from ip_src.data import GraphBuilder, InstantPolicyDataset
    print("✓ ip_src.data: OK")
except ImportError as e:
    print(f"✗ ip_src.data: {e}")

try:
    from ip_src.training import InstantPolicyTrainer
    print("✓ ip_src.training: OK")
except ImportError as e:
    print(f"✗ ip_src.training: {e}")

print("\n--- Quick Model Test ---")
try:
    from ip_src.models.encoders import sinusoidal_positional_encoding
    import torch
    
    # Test positional encoding
    delta_p = torch.randn(100, 3)
    encoded = sinusoidal_positional_encoding(delta_p, num_frequencies=10)
    print(f"✓ sinusoidal_positional_encoding: input {delta_p.shape} -> output {encoded.shape}")
    
    # Test encoder (small test)
    from ip_src.models.encoders import PointNetPlusPlusEncoder
    encoder = PointNetPlusPlusEncoder(out_channels=256, num_output_points=16, freeze=False)
    pos = torch.randn(2048, 3)
    features, positions = encoder(pos)
    print(f"✓ PointNetPlusPlusEncoder: input {pos.shape} -> features {features.shape}, positions {positions.shape}")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")

print("\n=== Environment check complete ===")
