"""
PyTorch Lightning Trainer for Instant Policy.

Implements the training loop with:
- Diffusion-based action prediction
- Proper learning rate scheduling
- Logging and checkpointing

Training follows paper recommendations:
- Learning rate: 1e-5 (larger rates cause instability)
- Optimizer: AdamW
- Geometry encoder: frozen

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, Any
import numpy as np

from ip_src.models.graph_diffusion import GraphDiffusion
from ip_src.training.losses import CombinedLoss


class InstantPolicyTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training Instant Policy.
    
    Handles:
    - Forward pass with diffusion
    - Loss computation
    - Optimizer and scheduler configuration
    - Logging
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
            self.model = GraphDiffusion(
                device=config.get('hardware', {}).get('device', 'cuda'),
                hidden_dim=model_config.get('graph_transformer', {}).get('hidden_dim', 256),
                num_heads=model_config.get('graph_transformer', {}).get('num_heads', 8),
                num_layers=model_config.get('graph_transformer', {}).get('num_layers', 4),
                dropout=model_config.get('graph_transformer', {}).get('dropout', 0.1),
                num_points=config.get('graph', {}).get('num_points', 2048),
                k_neighbors=config.get('graph', {}).get('k_neighbors', 16),
                prediction_horizon=model_config.get('action', {}).get('prediction_horizon', 8),
                action_dim=model_config.get('action', {}).get('action_dim', 9),
                num_train_timesteps=model_config.get('diffusion', {}).get('num_train_timesteps', 1000),
                num_inference_timesteps=model_config.get('diffusion', {}).get('num_inference_timesteps', 4),
                freeze_geometry_encoder=model_config.get('geometry_encoder', {}).get('freeze', True),
            )
        else:
            self.model = model
        
        # Create loss function
        loss_config = config.get('training', {}).get('loss', {})
        self.loss_fn = CombinedLoss(
            diffusion_weight=loss_config.get('diffusion_weight', 1.0),
            gripper_weight=loss_config.get('gripper_weight', 0.1),
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, batch: Dict) -> Dict:
        """Forward pass through the model."""
        return self.model(
            demo_pcds=batch['demo_pcds'],
            demo_poses=batch['demo_poses'],
            demo_grips=batch['demo_grips'],
            live_pcd=batch['live_pcd'],
            live_pose=batch['live_pose'],
            live_grip=batch['live_grip'],
            target_actions=batch['target_actions'],
        )
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch of training data
            batch_idx: Batch index
        
        Returns:
            Loss value
        """
        # Forward pass
        outputs = self(batch)
        
        # Compute loss
        loss_dict = self.loss_fn(
            noise_pred=outputs['noise_pred'],
            noise=outputs['noise'],
            grip_pred=outputs['grip_pred'],
            grip_target=batch['target_grips'],
        )
        
        # Log metrics
        self.log('train/loss', loss_dict['loss'], prog_bar=True)
        self.log('train/diffusion_loss', loss_dict['diffusion_loss'])
        self.log('train/gripper_loss', loss_dict['gripper_loss'])
        
        return loss_dict['loss']
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        
        Args:
            batch: Batch of validation data
            batch_idx: Batch index
        
        Returns:
            Loss value
        """
        # Forward pass
        outputs = self(batch)
        
        # Compute loss
        loss_dict = self.loss_fn(
            noise_pred=outputs['noise_pred'],
            noise=outputs['noise'],
            grip_pred=outputs['grip_pred'],
            grip_target=batch['target_grips'],
        )
        
        # Log metrics
        self.log('val/loss', loss_dict['loss'], prog_bar=True, sync_dist=True)
        self.log('val/diffusion_loss', loss_dict['diffusion_loss'], sync_dist=True)
        self.log('val/gripper_loss', loss_dict['gripper_loss'], sync_dist=True)
        
        return loss_dict['loss']
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        opt_config = self.config.get('training', {}).get('optimizer', {})
        sched_config = self.config.get('training', {}).get('scheduler', {})
        
        # Create optimizer
        # Paper uses AdamW with lr=1e-5
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_config.get('lr', 1e-5),
            weight_decay=opt_config.get('weight_decay', 1e-4),
            betas=tuple(opt_config.get('betas', [0.9, 0.999])),
        )
        
        # Create scheduler
        scheduler_name = sched_config.get('name', 'cosine')
        
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('training', {}).get('num_epochs', 100),
                eta_min=sched_config.get('min_lr', 1e-7),
            )
        elif scheduler_name == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=sched_config.get('min_lr', 1e-7) / opt_config.get('lr', 1e-5),
                total_iters=self.config.get('training', {}).get('num_epochs', 100),
            )
        elif scheduler_name == 'warmup_cosine':
            # Custom warmup + cosine scheduler
            warmup_steps = sched_config.get('warmup_steps', 1000)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / (
                    self.trainer.estimated_stepping_batches - warmup_steps
                )
                return 0.5 * (1 + np.cos(np.pi * progress))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                },
            }
        else:
            # No scheduler
            return optimizer
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Log epoch-level metrics
        avg_loss = self.trainer.callback_metrics.get('train/loss', 0)
        self.log('train/epoch_loss', avg_loss)
    
    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch."""
        # Log epoch-level metrics
        avg_loss = self.trainer.callback_metrics.get('val/loss', 0)
        self.log('val/epoch_loss', avg_loss)
    
    def get_model(self) -> GraphDiffusion:
        """Get the underlying GraphDiffusion model."""
        return self.model
    
    def save_model(self, path: str):
        """
        Save just the model weights (not training state).
        
        Args:
            path: Path to save model
        """
        self.model.save_checkpoint(path)
    
    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        device: str = "cuda",
    ) -> "InstantPolicyTrainer":
        """
        Load a pretrained trainer.
        
        Args:
            checkpoint_path: Path to checkpoint
            config: Optional config override
            device: Device to load on
        
        Returns:
            Loaded trainer
        """
        if config is None:
            config = {}
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config from checkpoint if not provided
        if 'hyper_parameters' in checkpoint:
            saved_config = checkpoint.get('hyper_parameters', {}).get('config', {})
            # Merge with provided config
            for key in saved_config:
                if key not in config:
                    config[key] = saved_config[key]
        
        # Create trainer
        trainer = cls(config)
        
        # Load state dict
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
        if self.ema_weights is None:
            # Initialize EMA weights
            self.ema_weights = {
                name: param.data.clone()
                for name, param in pl_module.model.named_parameters()
            }
        else:
            # Update EMA
            for name, param in pl_module.model.named_parameters():
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
