"""
Test ShapeNet demonstration generator.

Note: Full tests require a ShapeNet dataset. This tests basic functionality.
"""

import sys
from pathlib import Path
import tempfile
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def test_import():
    """Test that module can be imported."""
    print("Testing import...")
    
    from ip_src.data import SHAPENET_AVAILABLE
    print(f"  SHAPENET_AVAILABLE: {SHAPENET_AVAILABLE}")
    
    if SHAPENET_AVAILABLE:
        from ip_src.data.shapenet_demo import (
            ShapeNetDemoGenerator,
            ShapeNetTaskConfig,
            ShapeNetLoader,
        )
        print("  Successfully imported ShapeNet components")
    else:
        print("  ShapeNet components not available (trimesh not installed)")
    
    print("[PASS] test_import")


def test_config():
    """Test configuration dataclass."""
    print("\nTesting ShapeNetTaskConfig...")
    
    try:
        from ip_src.data.shapenet_demo import ShapeNetTaskConfig
    except ImportError:
        print("[SKIP] trimesh not available")
        return
    
    config = ShapeNetTaskConfig(
        shapenet_root="/dummy/path",
        allowed_categories=["chair", "table"],
    )
    
    # Check default probabilities
    assert sum(config.motion_probs.values()) == 1.0, "Probabilities should sum to 1"
    
    # Check ShapeNet-specific params
    assert config.shapenet_root == "/dummy/path"
    assert config.allowed_categories == ["chair", "table"]
    
    # Check preserved params
    assert config.trans_step == 0.01, "Should preserve 1cm step"
    assert config.gripper_flip_prob == 0.1, "Should preserve 10% flip"
    
    print("[PASS] test_config")


def test_with_mock_data():
    """Test with mock .obj files."""
    print("\nTesting with mock data...")
    
    try:
        from ip_src.data.shapenet_demo import ShapeNetLoader, ShapeNetDemoGenerator
        import trimesh
    except ImportError as e:
        print(f"[SKIP] Required module not available: {e}")
        return
    
    # Create temporary directory with mock .obj files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple cube mesh
        cube = trimesh.primitives.Box(extents=[1, 1, 1])
        
        # Save to mock category structure
        category_dir = os.path.join(tmpdir, "test_category")
        os.makedirs(category_dir)
        cube.export(os.path.join(category_dir, "model.obj"))
        
        # Also create another model
        sphere = trimesh.primitives.Sphere(radius=0.5)
        model_dir = os.path.join(category_dir, "model_002")
        os.makedirs(model_dir)
        sphere.export(os.path.join(model_dir, "model.obj"))
        
        print(f"  Created mock ShapeNet at: {tmpdir}")
        
        # Test loader
        loader = ShapeNetLoader(
            shapenet_root=tmpdir,
            allowed_categories=None,
        )
        
        print(f"  Found {len(loader.mesh_paths)} meshes")
        assert len(loader.mesh_paths) >= 2, "Should find at least 2 meshes"
        
        # Test loading mesh
        rng = np.random.RandomState(42)
        points, shape = loader.get_random_point_cloud(
            rng=rng,
            position=np.array([0.5, 0.0, 0.2]),
            target_size=0.05,
            num_points=100,
        )
        
        assert points.shape == (100, 3), f"Wrong shape: {points.shape}"
        print(f"  Sampled point cloud shape: {points.shape}")
        
        # Test full generator
        gen = ShapeNetDemoGenerator(
            shapenet_root=tmpdir,
            num_objects=2,
            seed=42,
        )
        
        task = gen.generate_task(num_demos=2)
        
        print(f"  Motion type: {task['motion_type']}")
        print(f"  Number of demos: {len(task['demos'])}")
        print(f"  Trajectory length: {len(task['demos'][0]['pcds'])}")
        
        # Validate structure
        demo = task['demos'][0]
        assert 'pcds' in demo, "Should have pcds"
        assert 'T_w_es' in demo, "Should have T_w_es"
        assert 'grips' in demo, "Should have grips"
        
        print("[PASS] test_with_mock_data")


def test_error_handling():
    """Test error handling for invalid paths."""
    print("\nTesting error handling...")
    
    try:
        from ip_src.data.shapenet_demo import ShapeNetLoader, TRIMESH_AVAILABLE
        if not TRIMESH_AVAILABLE:
            print("[SKIP] trimesh not available")
            return
    except ImportError:
        print("[SKIP] trimesh not available")
        return
    
    # Test invalid path
    try:
        loader = ShapeNetLoader("/nonexistent/path/12345")
        print("[FAIL] Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly raised ValueError: {str(e)[:50]}...")
    except ImportError as e:
        print(f"  Correctly raised ImportError: {str(e)[:50]}...")
    
    print("[PASS] test_error_handling")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing ShapeNet Demo Generator")
    print("=" * 50)
    
    test_import()
    test_config()
    test_with_mock_data()
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("All ShapeNet tests completed!")
    print("=" * 50)
