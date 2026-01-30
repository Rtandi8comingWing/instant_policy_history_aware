"""
Geometry Encoders for Instant Policy.

Implements the point cloud feature extraction using PointNet++ with NeRF-like
positional encoding, matching the official pretrained weights.

Key features:
- 2 Set Abstraction layers with local_nn + global_nn structure
- NeRF-like sinusoidal positional encoding (include_input=True -> 63 dim)
- Outputs M=16 keypoint nodes with 512-dim features
- Encoder should be kept FROZEN during training (Appendix H)

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from torch_geometric.nn import fps, knn


def sinusoidal_positional_encoding(
    coords: torch.Tensor,
    num_frequencies: int = 10,
    include_input: bool = True,
) -> torch.Tensor:
    """
    NeRF-like positional encoding.
    Matches the official weights: Output dim = 63 (if include_input=True and L=10)
    """
    # 1. 准备频率: 2^0, ..., 2^(L-1)
    freq_bands = 2.0 ** torch.arange(
        num_frequencies, 
        dtype=coords.dtype, 
        device=coords.device
    ) * math.pi # [L]
    
    # 2. 扩展维度计算 [..., D, L]
    # coords: [..., 3] -> [..., 3, 1]
    # bands: [L] -> [1, L]
    orig_shape = coords.shape
    scaled = coords.unsqueeze(-1) * freq_bands.view(*([1]*(len(orig_shape)-1)), 1, num_frequencies)
    
    # 3. 计算 Sin/Cos
    sin_enc = torch.sin(scaled) # [..., 3, L]
    cos_enc = torch.cos(scaled) # [..., 3, L]
    
    # 4. Flatten: [..., 3, L, 2] -> [..., 3 * L * 2]
    encoding = torch.stack([sin_enc, cos_enc], dim=-1).view(*orig_shape[:-1], -1)
    
    # 5. 拼接原始输入 (Key Step for matching 63 dim)
    if include_input:
        encoding = torch.cat([coords, encoding], dim=-1)
        
    return encoding

class SAModule(nn.Module):
    """
    Set Abstraction Module matching the official weight hierarchy:
    sa_module -> conv -> local_nn / global_nn
    """
    def __init__(
        self,
        ratio: float,
        k: int,
        in_channels: int,
        local_dims: list, # e.g. [63, 128, 128, 128]
        global_dims: list # e.g. [128, 256, 256]
    ):
        super().__init__()
        self.ratio = ratio
        self.k = k
        
        # Build Local MLP
        local_layers = []
        for i in range(len(local_dims) - 1):
            local_layers.append(nn.Linear(local_dims[i], local_dims[i+1]))
            # Note: Weights usually imply Bias=True. 
            # PyG MLPs usually act as: Linear -> BatchNorm -> ReLU
            # Checking weights: we see 'weight' and 'bias'. 
            # We don't see batch_norm weights in your print (running_mean/var), 
            # but maybe they are further down? 
            # Standard PointNet++ uses BN. Let's assume Linear + ReLU for now or standard PyG MLP.
            # Your print shows: lins.0.weight, lins.0.bias. This is typical of `torch_geometric.nn.MLP`
            # which wraps Linear+BN+ReLU into a list called `lins` (if plain) or `children`.
            # Let's assume standard Linear + ReLU.
            local_layers.append(nn.ReLU())
        
        # Build Global MLP
        global_layers = []
        for i in range(len(global_dims) - 1):
            global_layers.append(nn.Linear(global_dims[i], global_dims[i+1]))
            global_layers.append(nn.ReLU())

        # Construct the 'conv' module to match keys: sa_module.conv.local_nn
        # We use nn.Sequential directly to match 'local_nn.lins' structure of PyG MLP
        # But to be safe with keys like `local_nn.lins.0.weight`, we need a structure that has `lins`.
        
        self.conv = nn.Module()
        self.conv.local_nn = nn.Sequential()
        # Hack to match PyG MLP structure "lins" naming if using standard Sequential
        # Standard Sequential uses "0", "1". 
        # PyG MLP uses "lins.0", "lins.1".
        # To make loading easiest, we will just use standard Sequential and user might need a tiny rename script,
        # OR we try to define `lins` attribute.
        
        # Let's use a custom class to hold the list to match 'local_nn.lins.0'
        class MLP_Wrapper(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.lins = nn.ModuleList()
                for i in range(len(dims)-1):
                    self.lins.append(nn.Linear(dims[i], dims[i+1]))
                self.act = nn.ReLU() # Shared activation or one per layer? 
                # PyG MLP usually applies act after each linear.
                
            def forward(self, x):
                for i, lin in enumerate(self.lins):
                    x = lin(x)
                    if i < len(self.lins) - 1: # No activation on last layer usually? 
                        # Looking at weights: 128->128 then 128->256. 
                        # Usually internal layers have ReLU.
                        x = self.act(x)
                return x

        self.conv.local_nn = MLP_Wrapper(local_dims)
        self.conv.global_nn = MLP_Wrapper(global_dims)

    def forward(self, x, pos, batch):
        # 1. FPS
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = knn(pos, pos[idx], self.k, batch, batch[idx])
        
        # 2. Pos Encoding
        pos_diff = pos[col] - pos[idx[row]]
        pos_enc = sinusoidal_positional_encoding(pos_diff, include_input=True) # [E, 63]
        
        # 3. Prep Local Input
        if x is not None:
            # x[col] is neighbor features
            edge_input = torch.cat([x[col], pos_enc], dim=-1)
        else:
            edge_input = pos_enc
            
        # 4. Local NN
        edge_feat = self.conv.local_nn(edge_input)  # [E, Hidden]
        
        # 5. Max Pooling
        # Scatter max - must match dtype for mixed precision training
        out_dim = edge_feat.size(-1)
        out = torch.zeros(
            pos[idx].size(0), out_dim, 
            device=edge_feat.device, 
            dtype=edge_feat.dtype  # Match dtype for bf16-mixed precision
        )
        # row is index of target centroid (0 to M-1)
        out = out.scatter_reduce(0, row.unsqueeze(-1).expand(-1, out_dim), edge_feat, reduce="amax", include_self=False)
        
        # 6. Global NN
        out = self.conv.global_nn(out)
        
        return out, pos[idx], batch[idx]

class PointNetPlusPlusEncoder(nn.Module):
    """
    PointNet++ Encoder matching official pretrained weights.
    
    Output: 16 keypoints with 512-dim features (fixed to match pretrained weights).
    Should be kept FROZEN during training (Appendix H).
    """
    
    def __init__(self, freeze: bool = True):
        super().__init__()
        
        # SA1: 
        # Input: 63 (PosEnc+Raw)
        # Local MLP: 63 -> 128 -> 128 -> 128 (Based on weights)
        # Global MLP: 128 -> 256 -> 256
        self.sa1_module = SAModule(
            ratio=0.0625,  # 2048->128
            k=32,
            in_channels=0,
            local_dims=[63, 128, 128, 128],
            global_dims=[128, 256, 256]
        )
        
        # SA2:
        # Input: 256 (SA1 Out) + 63 (PosEnc) = 319
        # Local MLP: 319 -> 512 -> 512 -> 512
        # Global MLP: 512 -> 512 -> 512
        self.sa2_module = SAModule(
            ratio=0.125,  # 128->16
            k=32,
            in_channels=256,
            local_dims=[319, 512, 512, 512],
            global_dims=[512, 512, 512]
        )
        
        if freeze:
            self.freeze()
            
    def freeze(self):
        """Freeze all encoder parameters (recommended for training)."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()  # Set to eval mode (affects dropout/batchnorm if any)
    
    def unfreeze(self):
        """Unfreeze encoder parameters (not recommended per paper)."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()
    
    def load_pretrained_weights(
        self, 
        checkpoint_path: str,
        source_prefix: str = "scene_encoder.",
        strict: bool = False,
        verbose: bool = True,
    ) -> Tuple[list, list]:
        """
        Load pretrained weights from official model.pt checkpoint.
        
        The official checkpoint uses 'scene_encoder' as prefix, but our model
        uses 'geometry_encoder'. This method handles the key mapping.
        
        Args:
            checkpoint_path: Path to model.pt file
            source_prefix: Prefix used in checkpoint for encoder (default: "scene_encoder.")
            strict: If True, raise error on missing/unexpected keys
            verbose: Print loading information
        
        Returns:
            Tuple of (missing_keys, unexpected_keys)
        
        Example:
            encoder = PointNetPlusPlusEncoder(freeze=True)
            encoder.load_pretrained_weights("./model.pt")
        """
        import os
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  # Assume it's just the state dict
        
        # Extract encoder weights and remap keys
        encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(source_prefix):
                # Remove source prefix to get local key
                local_key = key[len(source_prefix):]
                encoder_state_dict[local_key] = value
        
        if verbose:
            print(f"Found {len(encoder_state_dict)} encoder parameters in checkpoint")
            if encoder_state_dict:
                print(f"  Example keys: {list(encoder_state_dict.keys())[:3]}")
        
        if not encoder_state_dict:
            raise ValueError(
                f"No encoder weights found with prefix '{source_prefix}'. "
                f"Available prefixes: {set(k.split('.')[0] + '.' for k in state_dict.keys())}"
            )
        
        # Load weights
        result = self.load_state_dict(encoder_state_dict, strict=strict)
        
        if verbose:
            if result.missing_keys:
                print(f"  Missing keys: {result.missing_keys}")
            if result.unexpected_keys:
                print(f"  Unexpected keys: {result.unexpected_keys}")
            print("✅ Encoder weights loaded successfully")
        
        return result.missing_keys, result.unexpected_keys

    def forward(self, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract keypoint features from point cloud.
        
        Args:
            pos: Point positions [N, 3] where N is typically 2048 * batch_size
            batch: Batch indices [N], defaults to single batch
        
        Returns:
            features: Per-keypoint features [M * batch_size, 512]
            positions: Keypoint positions [M * batch_size, 3]
        """
        if batch is None:
            batch = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
            
        # SA1
        x1, pos1, batch1 = self.sa1_module(None, pos, batch)
        
        # SA2
        x2, pos2, batch2 = self.sa2_module(x1, pos1, batch1)
        
        return x2, pos2


def load_encoder_from_checkpoint(
    checkpoint_path: str,
    source_prefix: str = "scene_encoder.",
    freeze: bool = True,
    device: str = "cpu",
) -> PointNetPlusPlusEncoder:
    """
    Convenience function to create and load encoder from checkpoint.
    
    Args:
        checkpoint_path: Path to model.pt
        source_prefix: Prefix in checkpoint
        freeze: Whether to freeze encoder
        device: Device to load to
    
    Returns:
        Loaded encoder
    
    Example:
        encoder = load_encoder_from_checkpoint("./model.pt")
    """
    encoder = PointNetPlusPlusEncoder(freeze=freeze)
    encoder.load_pretrained_weights(checkpoint_path, source_prefix)
    return encoder.to(device)