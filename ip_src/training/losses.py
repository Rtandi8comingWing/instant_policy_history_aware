"""
Loss Functions for Instant Policy Training.

Implements the loss functions for graph-based flow matching:
- Flow loss (MSE between predicted and target flow vectors)
- Gripper loss (binary cross-entropy)

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FlowLoss(nn.Module):
    """
    Flow matching loss.
    
    Computes MSE between predicted and target flow (velocity) vectors.
    This is used instead of noise prediction in standard diffusion.
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        reduction: str = "mean",
    ):
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
        pred_flow: torch.Tensor,
        target_flow: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow loss.
        
        Args:
            pred_flow: Predicted flow [B, horizon, 6, 3]
            target_flow: Target flow [B, horizon, 6, 3]
            mask: Optional mask [B, horizon]
        
        Returns:
            Loss value
        """
        loss = self.loss_fn(pred_flow, target_flow)
        
        if mask is not None and self.reduction == "none":
            # Expand mask to match flow dimensions
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # [B, horizon, 1, 1]
            loss = loss * mask
            loss = loss.sum() / (mask.sum() * pred_flow.shape[-2] * pred_flow.shape[-1] + 1e-8)
        
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
            grip_pred: Predicted gripper logits [B, horizon, ...] 
            grip_target: Target gripper states [B, horizon] (0 or 1)
            mask: Optional mask [B, horizon]
        
        Returns:
            Loss value
        """
        # Flatten extra dimensions if present
        if grip_pred.dim() > 2:
            # Average across gripper nodes: [B, horizon, 6, 1] -> [B, horizon]
            grip_pred = grip_pred.mean(dim=tuple(range(2, grip_pred.dim())))
        
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
    
    Combines flow loss and gripper loss with configurable weights.
    """
    
    def __init__(
        self,
        flow_weight: float = 1.0,
        gripper_weight: float = 0.1,
        flow_loss_type: str = "mse",
    ):
        super().__init__()
        self.flow_weight = flow_weight
        self.gripper_weight = gripper_weight
        
        self.flow_loss = FlowLoss(loss_type=flow_loss_type)
        self.gripper_loss = GripperLoss()
    
    def forward(
        self,
        pred_flow: torch.Tensor,
        target_flow: torch.Tensor,
        pred_grip: torch.Tensor,
        target_grip: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_flow: Predicted flow [B, horizon, 6, 3]
            target_flow: Target flow [B, horizon, 6, 3]
            pred_grip: Predicted gripper logits [B, horizon, 6, 1]
            target_grip: Target gripper states [B, horizon]
            mask: Optional mask [B, horizon]
        
        Returns:
            Dictionary with individual and total losses
        """
        # Compute individual losses
        flow_loss = self.flow_loss(pred_flow, target_flow, mask)
        
        grip_loss = torch.tensor(0.0, device=pred_flow.device)
        if target_grip is not None:
            grip_loss = self.gripper_loss(pred_grip, target_grip, mask)
        
        # Weighted sum
        total_loss = (
            self.flow_weight * flow_loss +
            self.gripper_weight * grip_loss
        )
        
        return {
            'loss': total_loss,
            'flow_loss': flow_loss,
            'gripper_loss': grip_loss,
        }


# Backward compatibility alias
DiffusionLoss = FlowLoss
