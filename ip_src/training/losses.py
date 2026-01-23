"""
Loss Functions for Instant Policy Training.

Implements the loss functions from the paper:
- Diffusion loss (noise prediction MSE)
- Gripper loss (binary cross-entropy)

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class DiffusionLoss(nn.Module):
    """
    Diffusion loss for action prediction.
    
    Computes MSE between predicted and actual noise.
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean",
    ):
        """
        Initialize diffusion loss.
        
        Args:
            loss_type: Type of loss ("mse", "l1", "huber")
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction=reduction)
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diffusion loss.
        
        Args:
            noise_pred: Predicted noise [B, horizon, action_dim]
            noise: Ground truth noise [B, horizon, action_dim]
            mask: Optional mask for valid predictions [B, horizon]
        
        Returns:
            Loss value
        """
        loss = self.loss_fn(noise_pred, noise)
        
        if mask is not None and self.reduction == "none":
            # Apply mask and reduce
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / (mask.sum() * noise.shape[-1] + 1e-8)
        
        return loss


class GripperLoss(nn.Module):
    """
    Gripper state prediction loss.
    
    Binary cross-entropy for open/close prediction.
    """
    
    def __init__(
        self,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize gripper loss.
        
        Args:
            reduction: Reduction method
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self,
        grip_pred: torch.Tensor,
        grip_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute gripper loss.
        
        Args:
            grip_pred: Predicted gripper logits [B, horizon, 1] or [B, horizon]
            grip_target: Target gripper states [B, horizon] (0 or 1)
            mask: Optional mask [B, horizon]
        
        Returns:
            Loss value
        """
        # Flatten predictions if needed
        if grip_pred.dim() == 3:
            grip_pred = grip_pred.squeeze(-1)
        
        # Convert target to float
        grip_target = grip_target.float()
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            grip_target = grip_target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Compute BCE loss
        loss = F.binary_cross_entropy_with_logits(
            grip_pred, grip_target, reduction="none"
        )
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                loss = loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == "sum":
                loss = loss.sum()
        else:
            if self.reduction == "mean":
                loss = loss.mean()
            elif self.reduction == "sum":
                loss = loss.sum()
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for Instant Policy training.
    
    Combines diffusion loss and gripper loss with configurable weights.
    """
    
    def __init__(
        self,
        diffusion_weight: float = 1.0,
        gripper_weight: float = 0.1,
        diffusion_loss_type: str = "mse",
    ):
        """
        Initialize combined loss.
        
        Args:
            diffusion_weight: Weight for diffusion loss
            gripper_weight: Weight for gripper loss
            diffusion_loss_type: Type of diffusion loss
        """
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.gripper_weight = gripper_weight
        
        self.diffusion_loss = DiffusionLoss(loss_type=diffusion_loss_type)
        self.gripper_loss = GripperLoss()
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        grip_pred: torch.Tensor,
        grip_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            noise_pred: Predicted noise [B, horizon, action_dim]
            noise: Ground truth noise [B, horizon, action_dim]
            grip_pred: Predicted gripper logits [B, horizon, 1]
            grip_target: Target gripper states [B, horizon]
            mask: Optional mask [B, horizon]
        
        Returns:
            Dictionary with individual and total losses
        """
        # Compute individual losses
        diff_loss = self.diffusion_loss(noise_pred, noise, mask)
        grip_loss = self.gripper_loss(grip_pred, grip_target, mask)
        
        # Weighted sum
        total_loss = (
            self.diffusion_weight * diff_loss +
            self.gripper_weight * grip_loss
        )
        
        return {
            'loss': total_loss,
            'diffusion_loss': diff_loss,
            'gripper_loss': grip_loss,
        }


class ActionLoss(nn.Module):
    """
    Direct action prediction loss (for non-diffusion training).
    
    Separates translation and rotation components with different weights.
    """
    
    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.reduction = reduction
    
    def forward(
        self,
        action_pred: torch.Tensor,
        action_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute action loss.
        
        Args:
            action_pred: Predicted actions [B, horizon, 9]
            action_target: Target actions [B, horizon, 9]
        
        Returns:
            Dictionary with losses
        """
        # Split into translation and rotation
        trans_pred = action_pred[..., :3]
        trans_target = action_target[..., :3]
        
        rot_pred = action_pred[..., 3:9]
        rot_target = action_target[..., 3:9]
        
        # Compute losses
        trans_loss = F.mse_loss(trans_pred, trans_target, reduction=self.reduction)
        rot_loss = F.mse_loss(rot_pred, rot_target, reduction=self.reduction)
        
        total_loss = (
            self.translation_weight * trans_loss +
            self.rotation_weight * rot_loss
        )
        
        return {
            'loss': total_loss,
            'translation_loss': trans_loss,
            'rotation_loss': rot_loss,
        }
