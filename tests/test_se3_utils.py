"""
Tests for SE(3) utility functions.

Verifies the correctness of Logmap/Expmap operations.
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ip_src.utils.se3_utils import (
    se3_log_map,
    se3_exp_map,
    so3_log_map,
    so3_exp_map,
    SE3ActionNormalizer,
    compute_relative_transform,
    apply_transform_to_points,
    skew_symmetric,
)


def test_skew_symmetric():
    """Test skew-symmetric matrix construction."""
    v = torch.tensor([1.0, 2.0, 3.0])
    K = skew_symmetric(v)
    
    expected = torch.tensor([
        [0.0, -3.0, 2.0],
        [3.0, 0.0, -1.0],
        [-2.0, 1.0, 0.0]
    ])
    
    assert torch.allclose(K, expected), f"Expected {expected}, got {K}"
    print("[PASS] test_skew_symmetric")


def test_so3_exp_log_roundtrip():
    """Test that exp(log(R)) = R for SO(3)."""
    # Create a random rotation matrix using axis-angle
    axis = torch.tensor([1.0, 0.0, 0.0])  # Rotate around x-axis
    angle = 0.5  # radians
    omega = axis * angle
    
    # Exp map
    R = so3_exp_map(omega.unsqueeze(0)).squeeze(0)
    
    # Verify R is a valid rotation matrix
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5), "R is not orthogonal"
    assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-5), "det(R) != 1"
    
    # Log map
    omega_recovered = so3_log_map(R.unsqueeze(0)).squeeze(0)
    
    # Check roundtrip
    assert torch.allclose(omega, omega_recovered, atol=1e-5), \
        f"Expected {omega}, got {omega_recovered}"
    
    # Double-check by going back through exp
    R_recovered = so3_exp_map(omega_recovered.unsqueeze(0)).squeeze(0)
    assert torch.allclose(R, R_recovered, atol=1e-5), \
        f"R recovery failed"
    
    print("[PASS] test_so3_exp_log_roundtrip")


def test_se3_exp_log_roundtrip():
    """Test that exp(log(T)) = T for SE(3)."""
    # Create a SE(3) matrix
    # Rotation: 30 degrees around z-axis
    angle = np.pi / 6
    R = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0]
    ])
    t = torch.tensor([0.01, 0.02, 0.03])  # Small translation
    
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    # Log map
    xi = se3_log_map(T.unsqueeze(0)).squeeze(0)
    
    # Exp map
    T_recovered = se3_exp_map(xi.unsqueeze(0)).squeeze(0)
    
    # Check roundtrip
    assert torch.allclose(T, T_recovered, atol=1e-5), \
        f"SE(3) roundtrip failed\nExpected:\n{T}\nGot:\n{T_recovered}"
    
    print("[PASS] test_se3_exp_log_roundtrip")


def test_se3_identity():
    """Test that log(I) = 0 and exp(0) = I."""
    I = torch.eye(4).unsqueeze(0)
    
    # Log of identity
    xi = se3_log_map(I).squeeze(0)
    assert torch.allclose(xi, torch.zeros(6), atol=1e-5), \
        f"log(I) should be 0, got {xi}"
    
    # Exp of zero
    T = se3_exp_map(torch.zeros(1, 6)).squeeze(0)
    assert torch.allclose(T, torch.eye(4), atol=1e-5), \
        f"exp(0) should be I, got {T}"
    
    print("[PASS] test_se3_identity")


def test_action_normalizer():
    """Test SE(3) action normalization."""
    normalizer = SE3ActionNormalizer(trans_scale=0.01, rot_scale=0.05)
    
    # Create a small action (1cm translation, 3 degrees rotation)
    xi = torch.tensor([0.01, 0.005, -0.01, 0.05, 0.0, -0.05])
    
    # Normalize
    xi_norm = normalizer.normalize(xi.unsqueeze(0)).squeeze(0)
    
    # Check that it's in [-1, 1]
    assert xi_norm.abs().max() <= 1.0, f"Normalized value out of range: {xi_norm}"
    
    # Unnormalize
    xi_recovered = normalizer.unnormalize(xi_norm.unsqueeze(0)).squeeze(0)
    
    # Note: Due to clamping, values at the boundary may be clamped
    # For this test, the values should be within scale, so roundtrip should work
    assert torch.allclose(xi, xi_recovered, atol=1e-6), \
        f"Expected {xi}, got {xi_recovered}"
    
    print("[PASS] test_action_normalizer")


def test_compute_relative_transform():
    """Test relative transform computation T_EA = T_WE^{-1} @ T_WA."""
    # Create two poses
    T_WE = torch.eye(4)
    T_WE[:3, 3] = torch.tensor([1.0, 0.0, 0.0])  # At (1, 0, 0)
    
    T_WA = torch.eye(4)
    T_WA[:3, 3] = torch.tensor([1.1, 0.05, 0.0])  # At (1.1, 0.05, 0)
    
    # Compute relative
    T_EA = compute_relative_transform(
        T_WA.unsqueeze(0), 
        T_WE.unsqueeze(0)
    ).squeeze(0)
    
    # Expected: move (0.1, 0.05, 0) in world frame
    # Since T_WE has no rotation, T_EA should have same translation
    expected_t = torch.tensor([0.1, 0.05, 0.0])
    
    assert torch.allclose(T_EA[:3, 3], expected_t, atol=1e-5), \
        f"Expected translation {expected_t}, got {T_EA[:3, 3]}"
    
    # Verify: T_WE @ T_EA = T_WA
    T_WA_recovered = T_WE @ T_EA
    assert torch.allclose(T_WA, T_WA_recovered, atol=1e-5), \
        f"T_WE @ T_EA != T_WA"
    
    print("[PASS] test_compute_relative_transform")


def test_apply_transform_to_points():
    """Test point transformation."""
    # Points at origin
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    
    # Transform: translate by (1, 2, 3)
    T = torch.eye(4)
    T[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    
    transformed = apply_transform_to_points(
        points.unsqueeze(0),
        T.unsqueeze(0)
    ).squeeze(0)
    
    expected = points + torch.tensor([1.0, 2.0, 3.0])
    
    assert torch.allclose(transformed, expected, atol=1e-5), \
        f"Expected {expected}, got {transformed}"
    
    print("[PASS] test_apply_transform_to_points")


def test_batch_operations():
    """Test that batch operations work correctly."""
    batch_size = 4
    
    # Create batch of random SE(3) matrices
    # Use small rotations and translations
    xi_batch = torch.randn(batch_size, 6) * 0.1
    
    # Exp map
    T_batch = se3_exp_map(xi_batch)
    assert T_batch.shape == (batch_size, 4, 4), f"Wrong shape: {T_batch.shape}"
    
    # Log map
    xi_recovered = se3_log_map(T_batch)
    assert xi_recovered.shape == (batch_size, 6), f"Wrong shape: {xi_recovered.shape}"
    
    # Roundtrip
    T_recovered = se3_exp_map(xi_recovered)
    assert torch.allclose(T_batch, T_recovered, atol=1e-4), \
        "Batch roundtrip failed"
    
    print("[PASS] test_batch_operations")


def test_flow_matching_pipeline():
    """Test the complete flow matching pipeline for SE(3)."""
    # Simulate a training scenario
    horizon = 8
    
    # Create ground truth actions
    gt_xi = torch.randn(horizon, 6) * 0.05  # Small actions
    
    # Normalize
    normalizer = SE3ActionNormalizer()
    gt_xi_norm = normalizer.normalize(gt_xi)
    
    # Add noise (simulate noisy sample at t=0.5)
    t = 0.5
    noise = torch.randn_like(gt_xi_norm)
    noisy_xi_norm = (1 - t) * gt_xi_norm + t * noise
    
    # Target velocity
    target_velocity = noise - gt_xi_norm
    
    # Simulate model prediction (perfect prediction for test)
    pred_velocity = target_velocity
    
    # Denoising step
    denoised_xi_norm = noisy_xi_norm - t * pred_velocity
    
    # Should recover ground truth
    assert torch.allclose(denoised_xi_norm, gt_xi_norm, atol=1e-5), \
        "Flow matching denoising failed"
    
    # Unnormalize and Expmap
    denoised_xi = normalizer.unnormalize(denoised_xi_norm)
    T_actions = se3_exp_map(denoised_xi)
    
    assert T_actions.shape == (horizon, 4, 4), f"Wrong shape: {T_actions.shape}"
    
    print("[PASS] test_flow_matching_pipeline")


if __name__ == "__main__":
    print("Running SE(3) utility tests...\n")
    
    test_skew_symmetric()
    test_so3_exp_log_roundtrip()
    test_se3_exp_log_roundtrip()
    test_se3_identity()
    test_action_normalizer()
    test_compute_relative_transform()
    test_apply_transform_to_points()
    test_batch_operations()
    test_flow_matching_pipeline()
    
    print("\n" + "="*50)
    print("All SE(3) utility tests passed!")
    print("="*50)
