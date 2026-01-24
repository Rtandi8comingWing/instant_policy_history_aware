"""
Flow Matching for Instant Policy.

Implements geometric flow-based generation for action prediction.
Key difference from standard diffusion: outputs geometric flow vectors
[∇p_trans, ∇p_rot] instead of flat noise predictions.

The process works on ghost gripper node POSITIONS directly in 3D space,
using a flow matching formulation.

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class FlowMatchingConfig:
    """Configuration for flow matching process."""
    num_train_timesteps: int = 1000
    num_inference_timesteps: int = 4
    sigma_min: float = 0.001
    sigma_max: float = 1.0
    flow_type: str = "rectified"  # "rectified" or "variance_preserving"


class FlowMatchingScheduler(nn.Module):
    """
    Flow Matching Scheduler for geometric action generation.
    
    Uses rectified flow formulation where:
    - x_t = (1-t) * x_0 + t * x_1 (linear interpolation)
    - v = x_1 - x_0 (constant velocity)
    
    The model predicts velocity/flow vectors that move samples
    from noise (t=1) to target positions (t=0).
    """
    
    def __init__(self, config: Optional[FlowMatchingConfig] = None):
        super().__init__()
        
        if config is None:
            config = FlowMatchingConfig()
        
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.num_inference_timesteps = config.num_inference_timesteps
        
        # Compute timestep schedule
        self._set_timesteps(config.num_inference_timesteps)
    
    def _set_timesteps(self, num_steps: int):
        """Set inference timesteps."""
        # Linear spacing from 1 to 0
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
        self.register_buffer("inference_timesteps", timesteps)
        
        # Step sizes
        step_sizes = torch.ones(num_steps) / num_steps
        self.register_buffer("step_sizes", step_sizes)
    
    def set_num_inference_steps(self, num_steps: int):
        """Update inference steps."""
        self.num_inference_timesteps = num_steps
        self._set_timesteps(num_steps)
    
    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Interpolate between target (x_0) and noise (x_1) at time t.
        
        For rectified flow: x_t = (1-t) * x_0 + t * x_1
        - At t=0: x_t = x_0 (target)
        - At t=1: x_t = x_1 (noise)
        
        Args:
            x_0: Target positions [B, ...]
            x_1: Noise positions [B, ...]
            t: Timesteps [B] in range [0, 1]
        
        Returns:
            Interpolated positions [B, ...]
        """
        # Expand t for broadcasting
        while len(t.shape) < len(x_0.shape):
            t = t.unsqueeze(-1)
        
        if self.config.flow_type == "rectified":
            # Rectified flow: linear interpolation
            x_t = (1 - t) * x_0 + t * x_1
        else:
            # Variance-preserving flow
            alpha_t = 1 - t
            sigma_t = t
            x_t = alpha_t * x_0 + sigma_t * x_1
        
        return x_t
    
    def get_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute target velocity/flow from target (x_0) to noise (x_1).
        
        For rectified flow: v = x_1 - x_0 (constant velocity)
        
        During training, we predict this velocity v.
        During inference, we integrate backwards: x_{t-dt} = x_t - dt * v
        
        Args:
            x_0: Target positions [B, ...]
            x_1: Noise positions [B, ...]
            t: Timesteps [B] (unused for rectified flow)
        
        Returns:
            Target velocity [B, ...]
        """
        # Rectified flow: constant velocity
        return x_1 - x_0


# === Utility Functions ===

def compute_flow_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Compute flow matching loss.
    
    Args:
        pred_velocity: Predicted velocity [B, N, 3]
        target_velocity: Target velocity [B, N, 3]
        loss_type: Loss type ("mse", "l1", "huber")
    
    Returns:
        Loss value
    """
    if loss_type == "mse":
        return F.mse_loss(pred_velocity, target_velocity)
    elif loss_type == "l1":
        return F.l1_loss(pred_velocity, target_velocity)
    elif loss_type == "huber":
        return F.huber_loss(pred_velocity, target_velocity)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class PositionToTransform:
    """
    Utility to convert ghost positions to SE(3) transforms.
    
    Converts the 6 ghost node positions back to a 4x4 transformation matrix.
    
    Note: GraphDiffusion uses SVD-based recovery (svd_se3_recovery) instead,
    which is more robust. This class is kept as an alternative method.
    """
    
    @staticmethod
    def positions_to_transform(positions: torch.Tensor) -> torch.Tensor:
        """
        Convert 6 gripper node positions to SE(3) transform.
        
        Args:
            positions: Ghost positions [6, 3]
        
        Returns:
            Transform [4, 4]
        """
        # Positions represent:
        # 0, 1: fingertip positions (define y-axis)
        # 2, 3: forward/backward (define x-axis)
        # 4, 5: up/down (define z-axis)
        
        # Compute center as average of all
        center = positions.mean(dim=0)  # [3]
        
        # Compute axes from position pairs
        y_axis = positions[0] - positions[1]  # Left - Right
        y_axis = y_axis / (torch.norm(y_axis) + 1e-8)
        
        x_axis = positions[2] - positions[3]  # Forward - Backward
        x_axis = x_axis / (torch.norm(x_axis) + 1e-8)
        
        # Orthogonalize
        x_axis = x_axis - (x_axis @ y_axis) * y_axis
        x_axis = x_axis / (torch.norm(x_axis) + 1e-8)
        
        # Compute z from cross product
        z_axis = torch.cross(x_axis, y_axis)
        z_axis = z_axis / (torch.norm(z_axis) + 1e-8)
        
        # Build transform
        T = torch.eye(4, device=positions.device, dtype=positions.dtype)
        T[:3, 0] = x_axis
        T[:3, 1] = y_axis
        T[:3, 2] = z_axis
        T[:3, 3] = center
        
        return T
    
    @staticmethod
    def batch_positions_to_transforms(
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert batch of ghost positions to transforms.
        
        Args:
            positions: [horizon, 6, 3]
        
        Returns:
            Transforms [horizon, 4, 4]
        """
        horizon = positions.shape[0]
        transforms = []
        
        for t in range(horizon):
            T = PositionToTransform.positions_to_transform(positions[t])
            transforms.append(T)
        
        return torch.stack(transforms)
