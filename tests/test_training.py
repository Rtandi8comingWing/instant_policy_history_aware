"""
Quick test to verify training pipeline works with SE(3) flow matching.
"""

import torch
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ip_src.models import GraphDiffusion
from ip_src.data import InstantPolicyDataset


def test_model_creation():
    """Test that model can be created."""
    print("Testing model creation...")
    
    # Use default parameters to avoid dimension mismatch
    model = GraphDiffusion(
        device='cpu',
    )
    
    # Check SE(3) normalizer is present
    assert hasattr(model, 'action_normalizer'), "Missing action_normalizer"
    assert hasattr(model, 'gripper_offsets'), "Missing gripper_offsets"
    
    print("[PASS] Model creation")
    return model


def test_dataset():
    """Test that dataset returns target_poses."""
    print("\nTesting dataset...")
    
    try:
        dataset = InstantPolicyDataset(
            data_dir='./data/pseudo_demos',
            num_demos=2,
            num_waypoints=10,
        )
    except Exception as e:
        print(f"[SKIP] Dataset not available: {e}")
        return None
    
    sample = dataset[0]
    
    # Check required keys
    required_keys = [
        'demo_pcds', 'demo_poses', 'demo_grips',
        'live_pcd', 'live_pose', 'live_grip',
        'target_poses', 'target_positions', 'target_grips'
    ]
    
    for key in required_keys:
        assert key in sample, f"Missing key: {key}"
    
    # Check shapes
    target_poses = sample['target_poses']
    target_positions = sample['target_positions']
    
    print(f"  target_poses shape: {target_poses.shape}")
    print(f"  target_positions shape: {target_positions.shape}")
    
    assert target_poses.shape[0] == target_positions.shape[0], "Horizon mismatch"
    assert target_poses.shape[1:] == torch.Size([4, 4]), "Wrong pose shape"
    
    print("[PASS] Dataset")
    return dataset


def test_forward_pass(model, dataset):
    """Test forward pass with a batch."""
    print("\nTesting forward pass...")
    
    if dataset is None:
        print("[SKIP] No dataset available")
        return
    
    # Create a small batch
    from ip_src.data.dataset import collate_fn
    batch_size = 2
    samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    batch = collate_fn(samples)
    
    # Move to CPU
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key]
    
    # Forward pass
    model.train()
    try:
        outputs = model.forward(
            demo_pcds=batch['demo_pcds'],
            demo_poses=batch['demo_poses'],
            demo_grips=batch['demo_grips'],
            live_pcd=batch['live_pcd'],
            live_pose=batch['live_pose'],
            live_grip=batch['live_grip'],
            target_poses=batch['target_poses'],
            target_positions=batch.get('target_positions'),
            target_grips=batch.get('target_grips'),
        )
        
        print(f"  pred_flow shape: {outputs['pred_flow'].shape}")
        print(f"  target_flow shape: {outputs['target_flow'].shape}")
        
        # Check shapes match
        assert outputs['pred_flow'].shape == outputs['target_flow'].shape, \
            "Flow shape mismatch"
        
        # Check 6D flow (3 trans + 3 rot)
        assert outputs['pred_flow'].shape[-1] == 6, \
            f"Expected 6D flow, got {outputs['pred_flow'].shape[-1]}D"
        
        print("[PASS] Forward pass")
        return outputs
        
    except Exception as e:
        print(f"[FAIL] Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_loss_computation(model, dataset):
    """Test that loss can be computed."""
    print("\nTesting loss computation...")
    
    if dataset is None:
        print("[SKIP] No dataset available")
        return
    
    from ip_src.data.dataset import collate_fn
    batch_size = 2
    samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    batch = collate_fn(samples)
    
    model.train()
    try:
        loss = model.training_step(batch, 0)
        print(f"  Total loss: {loss.item():.6f}")
        
        # Check loss is finite
        assert torch.isfinite(loss), "Loss is not finite"
        
        print("[PASS] Loss computation")
        return loss
        
    except Exception as e:
        print(f"[FAIL] Loss computation error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("Testing SE(3) Flow Matching Training Pipeline")
    print("=" * 50)
    
    model = test_model_creation()
    dataset = test_dataset()
    outputs = test_forward_pass(model, dataset)
    loss = test_loss_computation(model, dataset)
    
    print("\n" + "=" * 50)
    if dataset is not None and outputs is not None and loss is not None:
        print("All training pipeline tests passed!")
    else:
        print("Some tests were skipped or failed.")
    print("=" * 50)
