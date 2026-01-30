"""
SE(3) Utilities for Instant Policy.

Implements Lie group operations for SE(3) transformations following 
Appendix B of the Instant Policy paper:
- Logmap: SE(3) -> se(3) (matrix to 6-vector)
- Expmap: se(3) -> SE(3) (6-vector to matrix)
- Normalization for flow matching

The se(3) representation uses [translation, rotation] = [t_x, t_y, t_z, r_x, r_y, r_z]
where rotation is represented as axis-angle (rotation vector).

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    Create skew-symmetric matrix from 3D vector.
    
    [v]_x = [[0, -v_z, v_y],
             [v_z, 0, -v_x],
             [-v_y, v_x, 0]]
    
    Args:
        v: 3D vector [..., 3]
    
    Returns:
        Skew-symmetric matrix [..., 3, 3]
    """
    batch_shape = v.shape[:-1]
    device = v.device
    dtype = v.dtype
    
    zero = torch.zeros(batch_shape, device=device, dtype=dtype)
    
    # Extract components
    v_x = v[..., 0]
    v_y = v[..., 1]
    v_z = v[..., 2]
    
    # Build skew-symmetric matrix
    row0 = torch.stack([zero, -v_z, v_y], dim=-1)
    row1 = torch.stack([v_z, zero, -v_x], dim=-1)
    row2 = torch.stack([-v_y, v_x, zero], dim=-1)
    
    return torch.stack([row0, row1, row2], dim=-2)


def so3_exp_map(omega: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from so(3) (axis-angle) to SO(3) (rotation matrix).
    
    Uses Rodrigues' formula:
    R = I + sin(θ) * [ω]_x + (1 - cos(θ)) * [ω]_x^2
    
    where θ = ||ω|| and [ω]_x is the skew-symmetric matrix.
    
    Args:
        omega: Rotation vectors (axis-angle) [..., 3]
    
    Returns:
        Rotation matrices [..., 3, 3]
    """
    batch_shape = omega.shape[:-1]
    device = omega.device
    dtype = omega.dtype
    
    # Compute angle (norm of rotation vector)
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [..., 1]
    theta_sq = theta ** 2
    
    # Handle small angles with Taylor expansion
    small_angle = theta < 1e-6
    
    # Normalize axis (with safety for small angles)
    axis = omega / (theta + 1e-8)  # [..., 3]
    
    # Skew-symmetric matrix
    K = skew_symmetric(axis)  # [..., 3, 3]
    K_sq = torch.bmm(K.reshape(-1, 3, 3), K.reshape(-1, 3, 3)).reshape(*batch_shape, 3, 3)
    
    # Rodrigues' formula coefficients
    sin_theta = torch.sin(theta)  # [..., 1]
    cos_theta = torch.cos(theta)  # [..., 1]
    
    # Expand for matrix operations
    sin_theta = sin_theta.unsqueeze(-1)  # [..., 1, 1]
    cos_theta = cos_theta.unsqueeze(-1)  # [..., 1, 1]
    
    # R = I + sin(θ) * K + (1 - cos(θ)) * K^2
    I = torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3)
    R = I + sin_theta * K + (1 - cos_theta) * K_sq
    
    # For very small angles, use identity (or first-order approximation)
    small_angle_expanded = small_angle.unsqueeze(-1).expand(*batch_shape, 3, 3)
    R = torch.where(small_angle_expanded, I + skew_symmetric(omega), R)
    
    return R


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map from SO(3) (rotation matrix) to so(3) (axis-angle).
    
    Args:
        R: Rotation matrices [..., 3, 3]
    
    Returns:
        Rotation vectors (axis-angle) [..., 3]
    """
    batch_shape = R.shape[:-2]
    device = R.device
    dtype = R.dtype
    
    # Compute trace
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    # Compute angle: cos(θ) = (trace(R) - 1) / 2
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)  # [...]
    
    # Handle small angles
    small_angle = theta < 1e-6
    
    # For small angles: ω ≈ (R - R^T) / 2 (vee map)
    skew = (R - R.transpose(-1, -2)) / 2.0
    omega_small = torch.stack([
        skew[..., 2, 1],
        skew[..., 0, 2],
        skew[..., 1, 0]
    ], dim=-1)
    
    # For larger angles: ω = θ * axis
    # From Rodrigues: R - R^T = 2 * sin(θ) * K
    # So skew = (R - R^T) / 2 = sin(θ) * K
    # To get ω = θ * axis, we need: ω = (θ / sin(θ)) * vee(skew)
    sin_theta = torch.sin(theta)
    sin_theta_safe = torch.where(sin_theta.abs() < 1e-8, 
                                  torch.ones_like(sin_theta), 
                                  sin_theta)
    
    # Scale: θ / sin(θ) to recover ω from sin(θ) * K
    scale = theta / sin_theta_safe
    scale = scale.unsqueeze(-1)  # [..., 1]
    
    omega_large = scale * torch.stack([
        skew[..., 2, 1],
        skew[..., 0, 2],
        skew[..., 1, 0]
    ], dim=-1)
    
    # Handle angle close to π (need to extract axis from R)
    near_pi = theta > (np.pi - 0.01)
    if near_pi.any():
        # R ≈ I + 2 * n * n^T - I = 2 * n * n^T for θ ≈ π
        # So diagonal elements give n^2
        diag = torch.diagonal(R, dim1=-2, dim2=-1)  # [..., 3]
        # Find dominant axis
        # This is a simplified version - full implementation would be more robust
        pass
    
    # Select based on angle magnitude
    omega = torch.where(small_angle.unsqueeze(-1).expand_as(omega_small),
                       omega_small, omega_large)
    
    return omega


def se3_exp_map(xi: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from se(3) (twist) to SE(3) (transformation matrix).
    
    Uses the closed-form solution:
    T = [[R, V * t],
         [0, 1]]
    
    where R = exp([ω]_x) and V = I + (1-cos(θ))/θ^2 * [ω]_x + (θ-sin(θ))/θ^3 * [ω]_x^2
    
    Args:
        xi: Twist vectors [translation, rotation] = [..., 6]
            First 3 components: translation
            Last 3 components: rotation (axis-angle)
    
    Returns:
        SE(3) transformation matrices [..., 4, 4]
    """
    batch_shape = xi.shape[:-1]
    device = xi.device
    dtype = xi.dtype
    
    # Split into translation and rotation
    t = xi[..., :3]  # [..., 3] - translation
    omega = xi[..., 3:]  # [..., 3] - rotation (axis-angle)
    
    # Compute rotation matrix
    R = so3_exp_map(omega)  # [..., 3, 3]
    
    # Compute angle
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [..., 1]
    theta_sq = theta ** 2
    theta_cu = theta ** 3
    
    # Handle small angles
    small_angle = theta < 1e-6
    
    # V matrix coefficients (for transforming translation)
    # V = I + (1-cos(θ))/θ^2 * [ω]_x + (θ-sin(θ))/θ^3 * [ω]_x^2
    # where [ω]_x is the skew-symmetric matrix of omega (not normalized axis)
    
    # K is skew-symmetric of omega directly
    K = skew_symmetric(omega)  # [..., 3, 3]
    K_sq = torch.bmm(K.reshape(-1, 3, 3), K.reshape(-1, 3, 3)).reshape(*batch_shape, 3, 3)
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    # Coefficients with small angle handling
    # For small θ: (1-cos(θ))/θ^2 ≈ 1/2, (θ-sin(θ))/θ^3 ≈ 1/6
    theta_sq_safe = torch.where(theta_sq < 1e-8, torch.ones_like(theta_sq), theta_sq)
    theta_cu_safe = torch.where(theta_cu < 1e-8, torch.ones_like(theta_cu), theta_cu)
    
    c1 = (1 - cos_theta) / theta_sq_safe
    c2 = (theta - sin_theta) / theta_cu_safe
    
    c1 = torch.where(small_angle, 0.5 * torch.ones_like(c1), c1)
    c2 = torch.where(small_angle, (1.0/6.0) * torch.ones_like(c2), c2)
    
    c1 = c1.unsqueeze(-1)  # [..., 1, 1]
    c2 = c2.unsqueeze(-1)
    
    I = torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3)
    V = I + c1 * K + c2 * K_sq  # [..., 3, 3]
    
    # Transform translation: p = V * t
    p = torch.bmm(V.reshape(-1, 3, 3), t.unsqueeze(-1).reshape(-1, 3, 1))
    p = p.reshape(*batch_shape, 3)
    
    # Build SE(3) matrix
    T = torch.zeros(*batch_shape, 4, 4, device=device, dtype=dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    
    return T


def se3_log_map(T: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map from SE(3) (transformation matrix) to se(3) (twist).
    
    Args:
        T: SE(3) transformation matrices [..., 4, 4]
    
    Returns:
        Twist vectors [translation, rotation] = [..., 6]
            First 3 components: translation (not the direct t from T!)
            Last 3 components: rotation (axis-angle)
    """
    batch_shape = T.shape[:-2]
    device = T.device
    dtype = T.dtype
    
    # Extract rotation and translation
    R = T[..., :3, :3]  # [..., 3, 3]
    p = T[..., :3, 3]   # [..., 3]
    
    # Compute rotation (axis-angle)
    omega = so3_log_map(R)  # [..., 3]
    
    # Compute angle
    theta = torch.norm(omega, dim=-1, keepdim=True)  # [..., 1]
    theta_sq = theta ** 2
    theta_cu = theta ** 3
    
    # Handle small angles
    small_angle = theta < 1e-6
    
    # Compute V^{-1} to recover translation component
    # V^{-1} = I - 1/2 * [ω]_x + c * [ω]_x^2
    # where c = (1/θ^2 - (1+cos(θ))/(2θ*sin(θ)))
    # Note: [ω]_x = theta * [axis]_x, so we use omega directly
    
    # K is the skew-symmetric matrix of omega (not normalized axis)
    K = skew_symmetric(omega)  # [..., 3, 3]
    K_sq = torch.bmm(K.reshape(-1, 3, 3), K.reshape(-1, 3, 3)).reshape(*batch_shape, 3, 3)
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    # Coefficient for K^2 term with small angle handling
    # For large θ: c = 1/θ^2 - (1+cos(θ))/(2θ*sin(θ))
    # For small θ: c ≈ 1/12 (Taylor expansion)
    theta_sq_safe = torch.where(theta_sq < 1e-8, torch.ones_like(theta_sq), theta_sq)
    theta_safe = torch.where(theta < 1e-8, torch.ones_like(theta), theta)
    sin_theta_safe = torch.where(sin_theta.abs() < 1e-8, 
                                  torch.ones_like(sin_theta), sin_theta)
    
    c = 1.0 / theta_sq_safe - (1.0 + cos_theta) / (2.0 * theta_safe * sin_theta_safe)
    c = torch.where(small_angle, (1.0/12.0) * torch.ones_like(c), c)
    c = c.unsqueeze(-1)  # [..., 1, 1]
    
    I = torch.eye(3, device=device, dtype=dtype).expand(*batch_shape, 3, 3)
    V_inv = I - 0.5 * K + c * K_sq  # [..., 3, 3]
    
    # Recover translation: t = V^{-1} * p
    t = torch.bmm(V_inv.reshape(-1, 3, 3), p.unsqueeze(-1).reshape(-1, 3, 1))
    t = t.reshape(*batch_shape, 3)
    
    # Concatenate [translation, rotation]
    xi = torch.cat([t, omega], dim=-1)
    
    return xi


class SE3ActionNormalizer(nn.Module):
    """
    Normalizer for SE(3) actions in se(3) space.
    
    Following Appendix E of the paper:
    - Translation clipped to 1cm (0.01m)
    - Rotation clipped to 3 degrees (≈0.0524 radians)
    
    Normalizes to [-1, 1] range for stable flow matching.
    """
    
    def __init__(
        self,
        trans_scale: float = 0.01,  # 1cm
        rot_scale: float = 0.0524,  # ~3 degrees in radians
    ):
        """
        Initialize the normalizer.
        
        Args:
            trans_scale: Scale for translation normalization (in meters)
            rot_scale: Scale for rotation normalization (in radians)
        """
        super().__init__()
        
        # Action scale: [tx, ty, tz, rx, ry, rz]
        action_scale = torch.tensor([
            trans_scale, trans_scale, trans_scale,  # Translation
            rot_scale, rot_scale, rot_scale,        # Rotation
        ])
        self.register_buffer("action_scale", action_scale)
        
    def normalize(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Normalize se(3) vector to [-1, 1] range.
        
        Args:
            xi: se(3) vectors [..., 6]
        
        Returns:
            Normalized vectors [..., 6] in [-1, 1]
        """
        return torch.clamp(xi / self.action_scale, -1.0, 1.0)
    
    def unnormalize(self, xi_norm: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize from [-1, 1] to original se(3) scale.
        
        Args:
            xi_norm: Normalized vectors [..., 6]
        
        Returns:
            se(3) vectors [..., 6]
        """
        return xi_norm * self.action_scale


def compute_relative_transform(T_WA: torch.Tensor, T_WE: torch.Tensor) -> torch.Tensor:
    """
    Compute relative transformation T_EA = T_WE^{-1} @ T_WA.
    
    This is the action that transforms from current pose (E) to target pose (A)
    expressed in the end-effector frame.
    
    Args:
        T_WA: Target pose in world frame [..., 4, 4]
        T_WE: Current pose in world frame [..., 4, 4]
    
    Returns:
        Relative transform T_EA [..., 4, 4]
    """
    # Invert T_WE to get T_EW
    T_EW = torch.linalg.inv(T_WE)
    
    # T_EA = T_EW @ T_WA
    T_EA = torch.bmm(T_EW.reshape(-1, 4, 4), T_WA.reshape(-1, 4, 4))
    T_EA = T_EA.reshape(*T_WA.shape[:-2], 4, 4)
    
    return T_EA


def apply_transform_to_points(
    points: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    """
    Apply SE(3) transformation to 3D points.
    
    Args:
        points: 3D points [..., N, 3]
        transform: SE(3) transformation [..., 4, 4]
    
    Returns:
        Transformed points [..., N, 3]
    """
    # Extract rotation and translation
    R = transform[..., :3, :3]  # [..., 3, 3]
    t = transform[..., :3, 3]   # [..., 3]
    
    # Apply: p' = R @ p + t
    # For batch operation: [..., N, 3] @ [..., 3, 3]^T + [..., 1, 3]
    transformed = torch.einsum('...ij,...kj->...ik', points, R) + t.unsqueeze(-2)
    
    return transformed


def poses_to_gripper_positions(
    poses: torch.Tensor,
    gripper_offsets: torch.Tensor,
) -> torch.Tensor:
    """
    Convert SE(3) poses to 6 gripper node positions.
    
    Args:
        poses: SE(3) poses [..., 4, 4]
        gripper_offsets: Local gripper offsets [6, 3]
    
    Returns:
        Gripper positions [..., 6, 3]
    """
    batch_shape = poses.shape[:-2]
    device = poses.device
    dtype = poses.dtype
    
    # Extract rotation and translation
    R = poses[..., :3, :3]  # [..., 3, 3]
    t = poses[..., :3, 3]   # [..., 3]
    
    # Transform offsets: world_offsets = R @ offsets^T
    # offsets: [6, 3] -> need [3, 6] for matmul
    offsets = gripper_offsets.to(device=device, dtype=dtype)
    
    # R: [..., 3, 3], offsets^T: [3, 6] -> world_offsets: [..., 3, 6]
    world_offsets = torch.einsum('...ij,jk->...ik', R, offsets.T)  # [..., 3, 6]
    world_offsets = world_offsets.transpose(-1, -2)  # [..., 6, 3]
    
    # Add translation: positions = world_offsets + t
    positions = world_offsets + t.unsqueeze(-2)  # [..., 6, 3]
    
    return positions


# Default gripper offsets (same as in GraphBuilder and Dataset)
DEFAULT_GRIPPER_OFFSETS = torch.tensor([
    [0.0, 0.04, 0.0],   # Left finger tip
    [0.0, -0.04, 0.0],  # Right finger tip
    [0.05, 0.0, 0.0],   # Forward (x-axis)
    [-0.03, 0.0, 0.0],  # Backward
    [0.0, 0.0, 0.03],   # Up (z-axis)
    [0.0, 0.0, -0.03],  # Down
])
