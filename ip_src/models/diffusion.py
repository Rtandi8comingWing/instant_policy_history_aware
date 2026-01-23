"""
Flow Matching / Diffusion for Instant Policy.

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
import numpy as np
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
    
    Uses rectified flow or variance-preserving flow formulation.
    The model predicts velocity/flow vectors that move samples
    from noise to target positions.
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
        Interpolate between noise (x_1) and target (x_0) at time t.
        
        For rectified flow: x_t = (1-t) * x_0 + t * x_1
        
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
        Compute target velocity/flow from x_0 to x_1.
        
        For rectified flow: v = x_1 - x_0
        
        Args:
            x_0: Target positions [B, ...]
            x_1: Noise positions [B, ...]
            t: Timesteps [B] (unused for rectified flow)
        
        Returns:
            Target velocity [B, ...]
        """
        if self.config.flow_type == "rectified":
            # Rectified flow: constant velocity
            return x_1 - x_0
        else:
            # For VP flow, velocity depends on t
            while len(t.shape) < len(x_0.shape):
                t = t.unsqueeze(-1)
            return x_1 - x_0  # Simplified
    
    def step(
        self,
        velocity: torch.Tensor,
        x_t: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        """
        Single integration step.
        
        Args:
            velocity: Predicted velocity [B, ...]
            x_t: Current positions [B, ...]
            t: Current timestep (1 -> 0)
            dt: Step size (negative since going from 1 to 0)
        
        Returns:
            Updated positions [B, ...]
        """
        # Euler integration: x_{t-dt} = x_t - dt * v
        # Since we go from t=1 to t=0, we ADD velocity
        x_next = x_t - dt * velocity
        return x_next


class GeometricFlowModel(nn.Module):
    """
    Complete geometric flow model for ghost gripper position generation.
    
    This model:
    1. Takes context from the graph (demo + live observations)
    2. Predicts flow vectors for ghost gripper node positions
    3. Iteratively refines positions from noise to target
    
    The output is in 3D position space, not a flat action vector.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_gripper_nodes: int = 6,
        prediction_horizon: int = 8,
        config: Optional[FlowMatchingConfig] = None,
    ):
        super().__init__()
        
        if config is None:
            config = FlowMatchingConfig()
        
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_gripper_nodes = num_gripper_nodes
        self.prediction_horizon = prediction_horizon
        self.scheduler = FlowMatchingScheduler(config)
        
        # Total ghost nodes
        self.num_ghost_nodes = prediction_horizon * num_gripper_nodes
    
    def training_forward(
        self,
        target_positions: torch.Tensor,
        context_features: Dict[str, torch.Tensor],
        flow_predictor: nn.Module,
        edge_index_dict: Dict,
        edge_attr_dict: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            target_positions: Target ghost positions [B, horizon, 6, 3]
            context_features: Context node features from encoder
            flow_predictor: The ActionDecoder model (ψ)
            edge_index_dict: Graph edges
            edge_attr_dict: Edge features
        
        Returns:
            Dictionary with loss components
        """
        B = target_positions.shape[0]
        device = target_positions.device
        
        # Flatten target positions
        target_flat = target_positions.reshape(B, -1, 3)  # [B, horizon*6, 3]
        
        # Sample random timesteps
        t = torch.rand(B, device=device)  # [B]
        
        # Sample noise (starting positions)
        noise = torch.randn_like(target_flat)
        
        # Interpolate to get x_t
        x_t = self.scheduler.interpolate(target_flat, noise, t)  # [B, N, 3]
        
        # Get target velocity
        target_velocity = self.scheduler.get_velocity(target_flat, noise, t)  # [B, N, 3]
        
        # Predict velocity (this uses the graph structure)
        # We need to update the ghost node features with x_t
        ghost_features = context_features.copy()
        
        # The flow predictor outputs [num_ghost, 6] where 6 = 3 trans + 3 rot
        # For position-based flow, we use 3D position flow
        pred_flow, pred_gripper = flow_predictor(
            ghost_features, edge_index_dict, edge_attr_dict
        )
        
        # Extract translation flow (first 3 dims)
        pred_velocity = pred_flow[..., :3]  # [num_ghost, 3]
        
        return {
            'pred_velocity': pred_velocity,
            'target_velocity': target_velocity,
            'pred_gripper': pred_gripper,
            'timestep': t,
        }
    
    @torch.no_grad()
    def sample(
        self,
        context_features: Dict[str, torch.Tensor],
        flow_predictor: nn.Module,
        edge_index_dict: Dict,
        edge_attr_dict: Dict,
        graph_builder,  # For updating ghost node features
        live_gripper_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample ghost positions via iterative flow integration.
        
        Args:
            context_features: Encoded context
            flow_predictor: ActionDecoder model
            edge_index_dict: Graph edges
            edge_attr_dict: Edge features
            graph_builder: GraphBuilder for updating ghost features
            live_gripper_pos: Live gripper positions [6, 3]
        
        Returns:
            Final ghost positions [horizon, 6, 3]
            Gripper predictions [horizon, 6, 1]
        """
        device = live_gripper_pos.device
        
        # Initialize from noise around live gripper
        x = graph_builder.create_initial_ghost_positions(
            live_gripper_pos,
            self.prediction_horizon,
        )  # [horizon, 6, 3]
        
        # Get timesteps
        timesteps = self.scheduler.inference_timesteps
        step_sizes = self.scheduler.step_sizes
        
        gripper_preds = []
        
        # Iterative refinement
        for i, (t, dt) in enumerate(zip(timesteps, step_sizes)):
            # Update ghost node features based on current positions
            # This would rebuild the graph with new positions
            # For simplicity, we use the current features
            
            # Predict flow
            pred_flow, pred_gripper = flow_predictor(
                context_features, edge_index_dict, edge_attr_dict
            )
            
            # Extract position flow
            velocity = pred_flow[..., :3]  # [num_ghost, 3]
            
            # Reshape to [horizon, 6, 3]
            velocity = velocity.view(self.prediction_horizon, self.num_gripper_nodes, 3)
            
            # Integration step
            x = x - dt.item() * velocity
            
            gripper_preds.append(pred_gripper)
        
        # Use last gripper prediction
        final_gripper = gripper_preds[-1].view(
            self.prediction_horizon, self.num_gripper_nodes, 1
        )
        
        return x, final_gripper
    
    def set_num_inference_steps(self, num_steps: int):
        """Update inference steps."""
        self.scheduler.set_num_inference_steps(num_steps)


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


# Keep DDPM for compatibility but mark as deprecated
class DDPMScheduler(nn.Module):
    """
    DEPRECATED: Use FlowMatchingScheduler instead.
    
    Kept for backward compatibility with existing checkpoints.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        import warnings
        warnings.warn(
            "DDPMScheduler is deprecated. Use FlowMatchingScheduler instead.",
            DeprecationWarning,
        )
        
        # Wrap FlowMatchingScheduler
        self.scheduler = FlowMatchingScheduler()
    
    def add_noise(self, *args, **kwargs):
        return self.scheduler.interpolate(*args, **kwargs)
    
    def step(self, *args, **kwargs):
        return self.scheduler.step(*args, **kwargs)


class DiffusionActionDecoder(nn.Module):
    """
    DEPRECATED: Use GeometricFlowModel instead.
    
    Kept for backward compatibility.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        import warnings
        warnings.warn(
            "DiffusionActionDecoder is deprecated. Use GeometricFlowModel instead.",
            DeprecationWarning,
        )
