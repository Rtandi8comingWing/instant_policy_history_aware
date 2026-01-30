#!/usr/bin/env python3
"""
Simulation Deployment Script for Instant Policy (ip_src version).

This script loads the GraphDiffusion model from ip_src and evaluates it
in RLBench simulation environments.

Usage:
    # Basic usage with trained checkpoint
    python deploy_sim_ip.py --checkpoint checkpoints/last.ckpt --task_name plate_out
    
    # Custom settings
    python deploy_sim_ip.py --checkpoint checkpoints/best.ckpt --task_name phone_on_base --num_demos 3 --num_rollouts 20
    
    # Evaluate multiple tasks
    python deploy_sim_ip.py --checkpoint checkpoints/last.ckpt --task_name plate_out,phone_on_base,open_box --num_rollouts 10
    
    # With visualization
    python deploy_sim_ip.py --checkpoint checkpoints/last.ckpt --task_name plate_out --no_headless

Arguments:
    --checkpoint: Path to model checkpoint (.ckpt file from training)
    --task_name: Task name or comma-separated list of tasks
    --num_demos: Number of demonstrations for context
    --num_rollouts: Number of evaluation rollouts per task
    --restrict_rot: Whether to restrict rotation bounds (1=yes, 0=no)
    --no_headless: Run with visualization
    --diffusion_steps: Number of diffusion/flow matching steps
    --execution_horizon: Number of action steps per prediction

Weight Loading:
    The .ckpt file (PyTorch Lightning checkpoint) contains the COMPLETE model,
    including geometry_encoder. No separate model.pt is needed.

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ip_src.models import GraphDiffusion
from sim_utils_ip import rollout_model, batch_evaluate_tasks, TASK_NAMES


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Deploy Instant Policy model in RLBench simulation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Task settings
    parser.add_argument(
        '--task_name', 
        type=str, 
        default='plate_out',
        help='Task name (single) or comma-separated list of tasks'
    )
    parser.add_argument(
        '--list_tasks',
        action='store_true',
        help='List all available tasks and exit'
    )
    
    # Model settings
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to model checkpoint (.ckpt file). Contains complete model weights.'
    )
    parser.add_argument(
        '--diffusion_steps', 
        type=int, 
        default=4,
        help='Number of diffusion/flow matching inference steps'
    )
    
    # Evaluation settings
    parser.add_argument(
        '--num_demos', 
        type=int, 
        default=2,
        help='Number of demonstrations for context'
    )
    parser.add_argument(
        '--num_rollouts', 
        type=int, 
        default=10,
        help='Number of evaluation rollouts per task'
    )
    parser.add_argument(
        '--execution_horizon', 
        type=int, 
        default=8,
        help='Number of action steps to execute per prediction'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=30,
        help='Maximum action prediction steps per rollout'
    )
    parser.add_argument(
        '--num_traj_wp',
        type=int,
        default=10,
        help='Number of waypoints per demonstration context'
    )
    
    # Environment settings
    parser.add_argument(
        '--restrict_rot', 
        type=int, 
        default=1,
        choices=[0, 1],
        help='Restrict rotation bounds (1=yes, 0=no)'
    )
    parser.add_argument(
        '--no_headless',
        action='store_true',
        help='Run with visualization (default: headless)'
    )
    
    # Device settings
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for inference'
    )
    
    return parser.parse_args()


def load_model(args) -> GraphDiffusion:
    """
    Load GraphDiffusion model from Lightning checkpoint.
    
    The .ckpt file contains the complete model state (including geometry_encoder),
    so no separate encoder checkpoint is needed.
    
    Args:
        args: Command line arguments
    
    Returns:
        Loaded model ready for inference
    """
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    checkpoint_path = args.checkpoint
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please provide a valid .ckpt file path with --checkpoint"
        )
    
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get hyperparameters from checkpoint
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters'].copy()
        print(f"Found hyperparameters: {list(hparams.keys())}")
        
        # Override device
        hparams['device'] = device
        
        # IMPORTANT: Remove encoder_checkpoint to avoid redundant loading
        # The .ckpt already contains trained encoder weights
        hparams.pop('encoder_checkpoint', None)
    else:
        print("No hyperparameters found, using defaults")
        hparams = {'device': device}
    
    # Create model (without loading encoder from model.pt)
    model = GraphDiffusion(**hparams)
    
    # Load complete state dict from checkpoint
    # This includes ALL weights: geometry_encoder, local_encoder, 
    # context_aggregator, action_decoder, etc.
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded state dict with {len(state_dict)} keys")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    
    model = model.to(device)
    
    # Configure model for inference
    model.set_num_demos(args.num_demos)
    model.set_num_diffusion_steps(args.diffusion_steps)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  - Num demos: {args.num_demos}")
    print(f"  - Diffusion steps: {args.diffusion_steps}")
    
    return model


def main():
    """Main entry point."""
    args = parse_args()
    
    # List tasks if requested
    if args.list_tasks:
        print("\nAvailable tasks:")
        for task_name in sorted(TASK_NAMES.keys()):
            print(f"  - {task_name}")
        return
    
    # Parse task names
    task_names = [t.strip() for t in args.task_name.split(',')]
    
    # Validate task names
    for task_name in task_names:
        if task_name not in TASK_NAMES:
            print(f"Error: Unknown task '{task_name}'")
            print(f"Available tasks: {list(TASK_NAMES.keys())}")
            return
    
    print("=" * 60)
    print("Instant Policy Simulation Deployment (ip_src version)")
    print("=" * 60)
    print(f"\nTask(s): {', '.join(task_names)}")
    print(f"Num demos: {args.num_demos}")
    print(f"Num rollouts: {args.num_rollouts}")
    print(f"Execution horizon: {args.execution_horizon}")
    print(f"Restrict rotation: {bool(args.restrict_rot)}")
    print(f"Headless: {not args.no_headless}")
    print()
    
    # Load model
    try:
        model = load_model(args)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    
    if len(task_names) == 1:
        # Single task evaluation
        task_name = task_names[0]
        print(f"\nEvaluating task: {task_name}")
        
        success_rate = rollout_model(
            model=model,
            num_demos=args.num_demos,
            task_name=task_name,
            num_rollouts=args.num_rollouts,
            max_execution_steps=args.max_steps,
            execution_horizon=args.execution_horizon,
            num_traj_wp=args.num_traj_wp,
            headless=not args.no_headless,
            restrict_rot=bool(args.restrict_rot),
            verbose=True,
        )
        
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"\nTask: {task_name}")
        print(f"Success rate: {success_rate:.2%} ({int(success_rate * args.num_rollouts)}/{args.num_rollouts})")
        
    else:
        # Multi-task evaluation
        results = batch_evaluate_tasks(
            model=model,
            task_names=task_names,
            num_demos=args.num_demos,
            num_rollouts=args.num_rollouts,
            max_execution_steps=args.max_steps,
            execution_horizon=args.execution_horizon,
            num_traj_wp=args.num_traj_wp,
            headless=not args.no_headless,
            restrict_rot=bool(args.restrict_rot),
            verbose=True,
        )
        
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        for task_name, sr in results.items():
            if sr is not None:
                print(f"  {task_name}: {sr:.2%}")
            else:
                print(f"  {task_name}: FAILED")
        
        if valid_results:
            avg_sr = sum(valid_results.values()) / len(valid_results)
            print(f"\nAverage success rate: {avg_sr:.2%}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
