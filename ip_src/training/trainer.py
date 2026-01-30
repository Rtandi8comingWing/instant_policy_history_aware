"""
PyTorch Lightning Trainer for Instant Policy.

NOTE: GraphDiffusion already inherits from pl.LightningModule and has
built-in training_step, validation_step, and configure_optimizers.

This module provides an optional wrapper for additional training features
like EMA, or can be used directly with GraphDiffusion.

Recommended usage:
    # Option 1: Use GraphDiffusion directly (simpler)
    from ip_src.models import GraphDiffusion
    model = GraphDiffusion(...)
    trainer = pl.Trainer(...)
    trainer.fit(model, train_loader, val_loader)
    
    # Option 2: Use this wrapper for additional features
    from ip_src.training import InstantPolicyTrainer
    model = InstantPolicyTrainer(config)
    trainer = pl.Trainer(...)
    trainer.fit(model, train_loader, val_loader)

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Any
import numpy as np

from ip_src.models.graph_diffusion import GraphDiffusion


class InstantPolicyTrainer(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Instant Policy training.
    
    This is a thin wrapper around GraphDiffusion that:
    1. Creates the model from config
    2. Delegates forward/training to the model
    3. Provides additional training utilities
    
    Note: GraphDiffusion itself is already a LightningModule with full
    training support. Use this wrapper only if you need the config-based
    initialization or additional features.
    """
    
    def __init__(
        self,
        config: Dict,
        model: Optional[GraphDiffusion] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration dictionary
            model: Optional pre-initialized model
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.config = config
        
        # Create model if not provided
        if model is None:
            model_config = config.get('model', {})
            graph_config = config.get('graph', {})
            
            self.model = GraphDiffusion(
                device=config.get('hardware', {}).get('device', 'cuda'),
                hidden_dim=model_config.get('graph_transformer', {}).get('hidden_dim', 256),
                edge_dim=model_config.get('graph_transformer', {}).get('hidden_dim', 256),
                num_heads=model_config.get('graph_transformer', {}).get('num_heads', 8),
                num_layers=model_config.get('graph_transformer', {}).get('num_layers', 4),
                dropout=model_config.get('graph_transformer', {}).get('dropout', 0.1),
                num_points=graph_config.get('num_points', 2048),
                k_neighbors=graph_config.get('k_neighbors', 16),
                num_gripper_nodes=graph_config.get('num_ee_nodes', 6),
                prediction_horizon=model_config.get('action', {}).get('prediction_horizon', 8),
                num_train_timesteps=model_config.get('diffusion', {}).get('num_train_timesteps', 1000),
                num_inference_timesteps=model_config.get('diffusion', {}).get('num_inference_timesteps', 4),
                freeze_geometry_encoder=model_config.get('geometry_encoder', {}).get('freeze', True),
                learning_rate=config.get('training', {}).get('optimizer', {}).get('lr', 1e-5),
                weight_decay=config.get('training', {}).get('optimizer', {}).get('weight_decay', 0.01),
            )
        else:
            self.model = model
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass - delegates to model."""
        return self.model(
            demo_pcds=batch['demo_pcds'],
            demo_poses=batch['demo_poses'],
            demo_grips=batch['demo_grips'],
            live_pcd=batch['live_pcd'],
            live_pose=batch['live_pose'],
            live_grip=batch['live_grip'],
            target_positions=batch['target_positions'],
            target_grips=batch.get('target_grips'),
        )
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step - delegates to model's training_step."""
        return self.model.training_step(batch, batch_idx)
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step - delegates to model's validation_step."""
        return self.model.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """Configure optimizer - delegates to model."""
        return self.model.configure_optimizers()
    
    def get_model(self) -> GraphDiffusion:
        """Get the underlying GraphDiffusion model."""
        return self.model
    
    def save_model(self, path: str):
        """Save just the model weights."""
        self.model.save_checkpoint(path)
    
    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        device: str = "cuda",
    ) -> "InstantPolicyTrainer":
        """Load a pretrained trainer."""
        if config is None:
            config = {}
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'hyper_parameters' in checkpoint:
            saved_config = checkpoint.get('hyper_parameters', {}).get('config', {})
            for key in saved_config:
                if key not in config:
                    config[key] = saved_config[key]
        
        trainer = cls(config)
        
        if 'state_dict' in checkpoint:
            trainer.load_state_dict(checkpoint['state_dict'])
        
        return trainer


class EMACallback(pl.Callback):
    """
    Exponential Moving Average callback for model weights.
    
    Maintains an EMA of model weights for smoother predictions.
    """
    
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
    ):
        """Update EMA weights after each batch."""
        # Get the actual model (handle both wrapper and direct usage)
        if hasattr(pl_module, 'model'):
            model = pl_module.model
        else:
            model = pl_module
        
        if self.ema_weights is None:
            self.ema_weights = {
                name: param.data.clone()
                for name, param in model.named_parameters()
            }
        else:
            for name, param in model.named_parameters():
                if name in self.ema_weights:
                    self.ema_weights[name].mul_(self.decay).add_(
                        param.data, alpha=1 - self.decay
                    )
    
    def apply_ema(self, model: nn.Module):
        """Apply EMA weights to model."""
        if self.ema_weights is not None:
            for name, param in model.named_parameters():
                if name in self.ema_weights:
                    param.data.copy_(self.ema_weights[name])
    
    def restore_weights(self, model: nn.Module, original_weights: Dict):
        """Restore original weights."""
        for name, param in model.named_parameters():
            if name in original_weights:
                param.data.copy_(original_weights[name])
