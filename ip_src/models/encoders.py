"""
Geometry Encoders for Instant Policy.

Implements the point cloud feature extraction using PointNet++.
As stated in the paper (Appendix H), the geometry encoder should be kept frozen
during training - training from scratch or fine-tuning leads to worse performance.

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, knn


class SAModule(nn.Module):
    """
    Set Abstraction Module from PointNet++.
    
    Performs:
    1. Sampling: FPS to select representative points
    2. Grouping: k-NN to find local neighborhoods
    3. PointNet: Extract local features
    """
    
    def __init__(
        self,
        ratio: float,
        k: int,
        mlp_channels: list,
    ):
        """
        Initialize Set Abstraction module.
        
        Args:
            ratio: Sampling ratio for FPS
            k: Number of neighbors for k-NN
            mlp_channels: Channel sizes for MLP [in_channels, ..., out_channels]
        """
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.conv = PointNetConv(MLP(mlp_channels), add_self_loops=False)
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Point features [N, C] or None
            pos: Point positions [N, 3]
            batch: Batch indices [N]
        
        Returns:
            x: Updated features [M, C']
            pos: Sampled positions [M, 3]
            batch: Updated batch indices [M]
        """
        # Farthest Point Sampling
        idx = fps(pos, batch, ratio=self.ratio)
        
        # k-NN graph
        row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        edge_index = torch.stack([col, row], dim=0)
        
        # Apply PointNet convolution
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        
        pos, batch = pos[idx], batch[idx]
        
        return x, pos, batch


class GlobalSAModule(nn.Module):
    """
    Global Set Abstraction Module.
    
    Aggregates all points into a single global feature.
    """
    
    def __init__(self, mlp_channels: list):
        super().__init__()
        self.mlp = MLP(mlp_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Point features [N, C]
            pos: Point positions [N, 3]
            batch: Batch indices [N]
        
        Returns:
            x: Global features [B, C']
            pos: Zero positions [B, 3]
            batch: Batch indices [B]
        """
        x = self.mlp(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        
        # Create placeholder positions and batch indices
        pos = pos.new_zeros(x.size(0), 3)
        batch = torch.arange(x.size(0), device=batch.device)
        
        return x, pos, batch


class FPModule(nn.Module):
    """
    Feature Propagation Module from PointNet++.
    
    Propagates features from sampled points back to original points
    using distance-weighted interpolation.
    """
    
    def __init__(self, k: int, mlp_channels: list):
        """
        Initialize Feature Propagation module.
        
        Args:
            k: Number of neighbors for interpolation
            mlp_channels: Channel sizes for MLP
        """
        super().__init__()
        self.k = k
        self.mlp = MLP(mlp_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        x_skip: Optional[torch.Tensor],
        pos_skip: torch.Tensor,
        batch_skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Features from coarser level [M, C]
            pos: Positions from coarser level [M, 3]
            batch: Batch indices from coarser level [M]
            x_skip: Skip connection features [N, C'] or None
            pos_skip: Original positions [N, 3]
            batch_skip: Original batch indices [N]
        
        Returns:
            Interpolated features [N, C'']
        """
        # k-NN from skip positions to current positions
        row, col = knn(pos, pos_skip, self.k, batch, batch_skip)
        
        # Compute distances
        diff = pos_skip[row] - pos[col]
        squared_dist = (diff * diff).sum(dim=-1, keepdim=True)
        
        # Distance-weighted interpolation
        weights = 1.0 / (squared_dist + 1e-8)
        
        # Normalize weights per target point
        weights_sum = torch.zeros(pos_skip.size(0), 1, device=weights.device)
        weights_sum.scatter_add_(0, row.unsqueeze(-1), weights)
        weights = weights / (weights_sum[row] + 1e-8)
        
        # Interpolate features
        x_interp = torch.zeros(pos_skip.size(0), x.size(1), device=x.device)
        x_interp.scatter_add_(0, row.unsqueeze(-1).expand(-1, x.size(1)), weights * x[col])
        
        # Concatenate with skip features
        if x_skip is not None:
            x_interp = torch.cat([x_interp, x_skip], dim=1)
        
        return self.mlp(x_interp)


class PointNetPlusPlusEncoder(nn.Module):
    """
    PointNet++ Encoder for extracting geometric features from point clouds.
    
    This encoder is kept FROZEN during training as stated in the paper
    (Appendix H: "Training from scratch did not work at all, while 
    fine-tuning resulted in significantly worse performance").
    
    Architecture follows the standard PointNet++ with:
    - 3 Set Abstraction layers for hierarchical feature extraction
    - 3 Feature Propagation layers for per-point features
    """
    
    def __init__(
        self,
        out_channels: int = 256,
        freeze: bool = True,
    ):
        """
        Initialize PointNet++ encoder.
        
        Args:
            out_channels: Output feature dimension per point
            freeze: Whether to freeze encoder weights (should be True)
        """
        super().__init__()
        self.out_channels = out_channels
        
        # Set Abstraction layers (encoder)
        # SA1: N points -> N/4 points
        self.sa1 = SAModule(
            ratio=0.25,
            k=32,
            mlp_channels=[3 + 3, 64, 64, 128],  # pos + pos_local
        )
        
        # SA2: N/4 points -> N/16 points
        self.sa2 = SAModule(
            ratio=0.25,
            k=32,
            mlp_channels=[128 + 3, 128, 128, 256],
        )
        
        # SA3: N/16 points -> N/64 points
        self.sa3 = SAModule(
            ratio=0.25,
            k=32,
            mlp_channels=[256 + 3, 256, 256, 512],
        )
        
        # Feature Propagation layers (decoder)
        # FP3: N/64 -> N/16
        self.fp3 = FPModule(k=3, mlp_channels=[512 + 256, 256, 256])
        
        # FP2: N/16 -> N/4
        self.fp2 = FPModule(k=3, mlp_channels=[256 + 128, 256, 256])
        
        # FP1: N/4 -> N
        self.fp1 = FPModule(k=3, mlp_channels=[256 + 3, 256, out_channels])  # +3 for original positions
        
        # Freeze if specified
        if freeze:
            self.freeze()
    
    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all encoder parameters (not recommended)."""
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract per-point features from point cloud.
        
        Args:
            pos: Point positions [N, 3]
            batch: Batch indices [N], defaults to single batch
        
        Returns:
            Per-point features [N, out_channels]
        """
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        
        # Store for skip connections
        pos0, batch0 = pos, batch
        x0 = pos  # Use positions as initial features
        
        # Encoder
        x1, pos1, batch1 = self.sa1(None, pos0, batch0)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)
        
        # Decoder
        x2 = self.fp3(x3, pos3, batch3, x2, pos2, batch2)
        x1 = self.fp2(x2, pos2, batch2, x1, pos1, batch1)
        x0 = self.fp1(x1, pos1, batch1, pos0, pos0, batch0)  # Use pos as skip features
        
        return x0


class PointNetEncoder(nn.Module):
    """
    Simple PointNet encoder as a lighter alternative.
    
    Less powerful than PointNet++ but faster and simpler.
    """
    
    def __init__(
        self,
        out_channels: int = 256,
        freeze: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        
        # Per-point MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Global feature extraction
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        
        # Per-point output (combines local and global)
        self.mlp3 = nn.Sequential(
            nn.Linear(256 + 512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
        )
        
        if freeze:
            self.freeze()
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract per-point features.
        
        Args:
            pos: Point positions [N, 3]
            batch: Batch indices [N]
        
        Returns:
            Per-point features [N, out_channels]
        """
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        
        # Per-point features
        x = self.mlp1(pos)  # [N, 256]
        
        # Global features via max pooling
        x_global = self.mlp2(x)  # [N, 512]
        x_global = global_max_pool(x_global, batch)  # [B, 512]
        
        # Expand global features to all points
        x_global = x_global[batch]  # [N, 512]
        
        # Combine local and global
        x = torch.cat([x, x_global], dim=1)  # [N, 256 + 512]
        x = self.mlp3(x)  # [N, out_channels]
        
        return x


def load_pretrained_pointnet_plus_plus(
    checkpoint_path: Optional[str] = None,
    out_channels: int = 256,
) -> PointNetPlusPlusEncoder:
    """
    Load a pretrained PointNet++ encoder.
    
    Args:
        checkpoint_path: Path to pretrained weights (None for random init)
        out_channels: Output feature dimension
    
    Returns:
        Pretrained and frozen PointNet++ encoder
    """
    encoder = PointNetPlusPlusEncoder(out_channels=out_channels, freeze=True)
    
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        encoder.load_state_dict(state_dict, strict=False)
    
    return encoder
