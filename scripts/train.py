#!/usr/bin/env python3
"""
Training script for Instant Policy.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from ip_src.training.trainer import InstantPolicyTrainer
from ip_src.data.dataset import InstantPolicyDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Instant Policy")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/pseudo_demos",
        help="Path to pseudo-demonstration data",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = config["hardware"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = InstantPolicyDataset(
        data_dir=args.data_dir,
        num_points=config["graph"]["num_points"],
        num_waypoints=config["graph"]["num_waypoints"],
        num_demos=config["data"]["pseudo_demo"]["num_demos_per_task"],
    )
    
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        prefetch_factor=config["data"]["prefetch_factor"],
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    
    # Create model
    print("Creating model...")
    model = InstantPolicyTrainer(config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config["logging"]["checkpoint_dir"],
            filename="instant_policy-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Setup logger
    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=config["logging"]["project_name"],
            log_model=True,
        )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        precision=config["hardware"]["precision"] if device == "cuda" else 32,
        gradient_clip_val=config["training"]["gradient_clip"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        check_val_every_n_epoch=1,
    )
    
    # Train
    print("Starting training...")
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=args.resume,
    )
    
    print("Training complete!")
    print(f"Best checkpoint saved to: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
