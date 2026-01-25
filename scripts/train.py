#!/usr/bin/env python3
"""
Training script for Instant Policy.

Usage:
    # Generate pseudo-demonstrations first:
    python scripts/generate_pseudo_demos.py --output_dir ./data/pseudo_demos --num_tasks 1000
    
    # Then train:
    python scripts/train.py --config configs/default.yaml --data_dir ./data/pseudo_demos
    
    # Or train without wandb:
    python scripts/train.py --config configs/default.yaml --data_dir ./data/pseudo_demos --no_wandb
"""

import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from ip_src.models.graph_diffusion import GraphDiffusion
from ip_src.data.dataset import InstantPolicyDataset, collate_fn


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
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained model.pt for loading encoder weights",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = config["hardware"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Extract config sections
    graph_config = config.get('graph', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    # === Create Dataset ===
    print("Loading dataset...")
    try:
        dataset = InstantPolicyDataset(
            data_dir=args.data_dir,
            num_points=graph_config.get("num_raw_points", 2048),  # Raw point cloud size
            context_len=graph_config.get("num_waypoints", 10),    # L=10 per Appendix E
            num_demos=config["data"]["pseudo_demo"]["num_demos_per_task"],
            prediction_horizon=model_config.get("action", {}).get("prediction_horizon", 8),
        )
        print(f"Loaded {len(dataset)} tasks")
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease generate pseudo-demonstrations first:")
        print("  python scripts/generate_pseudo_demos.py --output_dir ./data/pseudo_demos --num_tasks 1000")
        print("Or generate ShapeNet-based demonstrations:")
        print("  python scripts/generate_shapenet_demos.py --shapenet_root ./data/shapenet --output_dir ./data/shapenet_demos --num_tasks 1000")
        return
    
    # Split into train/val
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        prefetch_factor=config["data"].get("prefetch_factor", 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # === Create Model ===
    print("Creating model...")
    
    # Get encoder checkpoint path (CLI arg > config > None)
    encoder_checkpoint = args.encoder_checkpoint
    if encoder_checkpoint is None:
        encoder_checkpoint = model_config.get('geometry_encoder', {}).get('pretrained_weights', None)
    
    model = GraphDiffusion(
        device=device,
        hidden_dim=model_config.get('graph_transformer', {}).get('hidden_dim', 256),
        edge_dim=model_config.get('graph_transformer', {}).get('hidden_dim', 256),
        num_heads=model_config.get('graph_transformer', {}).get('num_heads', 8),
        num_layers=model_config.get('graph_transformer', {}).get('num_layers', 4),
        dropout=model_config.get('graph_transformer', {}).get('dropout', 0.1),
        num_points=graph_config.get('num_keypoints', 16),  # Keypoints from geometry encoder, NOT raw points!
        k_neighbors=graph_config.get('k_neighbors', 8),
        num_gripper_nodes=graph_config.get('num_ee_nodes', 6),
        prediction_horizon=model_config.get('action', {}).get('prediction_horizon', 8),
        num_train_timesteps=model_config.get('diffusion', {}).get('num_train_timesteps', 1000),
        num_inference_timesteps=model_config.get('diffusion', {}).get('num_inference_timesteps', 4),
        freeze_geometry_encoder=model_config.get('geometry_encoder', {}).get('freeze', True),
        encoder_checkpoint=encoder_checkpoint,
        encoder_source_prefix=model_config.get('geometry_encoder', {}).get('source_prefix', 'scene_encoder.'),
        learning_rate=training_config.get('optimizer', {}).get('lr', 1e-5),
        weight_decay=training_config.get('optimizer', {}).get('weight_decay', 0.01),
    )
    
    # === Setup Callbacks ===
    callbacks = [
        ModelCheckpoint(
            dirpath=config["logging"]["checkpoint_dir"],
            filename="instant_policy-{epoch:02d}-{val/flow_loss:.4f}",
            save_top_k=3,
            monitor="val/flow_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # === Setup Logger ===
    logger = None
    if not args.no_wandb:
        try:
            logger = WandbLogger(
                project=config["logging"]["project_name"],
                log_model=True,
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            print("Continuing without wandb logging...")
    
    # === Create Trainer ===
    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        precision=config["hardware"].get("precision", "32") if device == "cuda" else 32,
        gradient_clip_val=config["training"].get("gradient_clip", 1.0),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        check_val_every_n_epoch=1,
    )
    
    # === Train ===
    print("Starting training...")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {training_config.get('optimizer', {}).get('lr', 1e-5)}")
    
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=args.resume,
    )
    
    print("\nTraining complete!")
    print(f"Best checkpoint saved to: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
