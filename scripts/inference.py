"""
Inference script for Instant Policy.

Loads a trained checkpoint and performs inference on pseudo-demo data
to verify model functionality.

Usage:
    python scripts/inference.py --checkpoint checkpoints/epoch=X-step=Y.ckpt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ip_src.models import GraphDiffusion
from ip_src.data import InstantPolicyDataset


def load_model(checkpoint_path: str, encoder_checkpoint: str = None, device: str = "cuda"):
    """Load model from Lightning checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get hyperparameters
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters'].copy()
        print(f"Found hyperparameters: {list(hparams.keys())}")
        # Override device with the one specified by user
        hparams['device'] = device
    else:
        print("No hyperparameters found, using defaults")
        hparams = {'device': device}
    
    # Create model
    model = GraphDiffusion(**hparams)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        # Filter out geometry_encoder if we're loading from pretrained
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded state dict with {len(state_dict)} keys")
    
    # Load encoder weights if provided separately
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        print(f"Loading encoder weights from: {encoder_checkpoint}")
        model.load_encoder_weights(encoder_checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def prepare_sample_from_dataset(dataset, sample_idx: int = 0):
    """
    Convert a dataset sample to the format expected by predict_actions.
    
    Dataset returns:
        - demo_pcds: [num_demos, context_len, num_points, 3] - Local EE frame
        - demo_poses: [num_demos, context_len, 4, 4] - World frame
        - demo_grips: [num_demos, context_len]
        - live_pcd: [num_points, 3] - Local EE frame
        - live_pose: [4, 4] - World frame
        - live_grip: scalar
    
    Returns:
        full_sample: Dict with 'demos' and 'live' keys
        raw_sample: Original dataset sample for ground truth comparison
    """
    sample = dataset[sample_idx]
    
    # Extract data - updated shapes per new dataset format
    demo_pcds = sample['demo_pcds'].numpy()      # [num_demos, context_len, N, 3]
    demo_poses = sample['demo_poses'].numpy()     # [num_demos, context_len, 4, 4]
    demo_grips = sample['demo_grips'].numpy()     # [num_demos, context_len]
    live_pcd = sample['live_pcd'].numpy()         # [N, 3]
    live_pose = sample['live_pose'].numpy()       # [4, 4]
    live_grip = sample['live_grip'].numpy()       # scalar or [1]
    
    num_demos, context_len = demo_pcds.shape[:2]
    
    # Build full_sample structure
    full_sample = {
        'demos': [],
        'live': {
            'obs': [live_pcd],
            'T_w_es': [live_pose],
            'grips': [float(live_grip)],
        }
    }
    
    for d in range(num_demos):
        demo_dict = {
            'obs': [demo_pcds[d, w] for w in range(context_len)],
            'T_w_es': [demo_poses[d, w] for w in range(context_len)],
            'grips': [float(demo_grips[d, w]) for w in range(context_len)],
        }
        full_sample['demos'].append(demo_dict)
    
    return full_sample, sample


def compute_predicted_positions(actions: np.ndarray, live_pose: np.ndarray) -> np.ndarray:
    """
    Compute predicted absolute positions from SE(3) actions.
    
    The actions are T_EA (action in end-effector frame), meaning:
    T_WA = T_WE @ T_EA
    
    Each action h represents the relative transform from live pose to target pose h,
    NOT incremental transforms between consecutive poses.
    
    Args:
        actions: Predicted SE(3) transforms [horizon, 4, 4] (T_EA in EE frame)
        live_pose: Live end-effector pose [4, 4] (T_WE)
    
    Returns:
        Predicted absolute positions [horizon, 3]
    """
    horizon = actions.shape[0]
    positions = []
    
    for h in range(horizon):
        # T_WA = T_WE @ T_EA
        # Each action is relative to the live pose, not the previous predicted pose
        T_WA = live_pose @ actions[h]
        positions.append(T_WA[:3, 3].copy())
    
    return np.array(positions)  # [horizon, 3]


def visualize_predictions(actions: np.ndarray, grips: np.ndarray, 
                          gt_positions: np.ndarray = None, gt_grips: np.ndarray = None,
                          live_pose: np.ndarray = None):
    """Visualize predicted vs ground truth actions."""
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    horizon = actions.shape[0]
    
    # Compute predicted absolute positions if live_pose is provided
    if live_pose is not None:
        pred_abs_positions = compute_predicted_positions(actions, live_pose)
    else:
        pred_abs_positions = None
    
    for h in range(horizon):
        print(f"\n--- Action Step {h+1}/{horizon} ---")
        
        # Extract translation from predicted SE(3)
        T_pred = actions[h]  # [4, 4]
        delta_trans = T_pred[:3, 3]
        
        print(f"  Predicted Delta: [{delta_trans[0]:.4f}, {delta_trans[1]:.4f}, {delta_trans[2]:.4f}]")
        
        if pred_abs_positions is not None:
            pred_pos = pred_abs_positions[h]
            print(f"  Predicted Abs:   [{pred_pos[0]:.4f}, {pred_pos[1]:.4f}, {pred_pos[2]:.4f}]")
        
        print(f"  Predicted Grip:  {'CLOSE' if grips[h] > 0.5 else 'OPEN'} ({grips[h, 0]:.3f})")
        
        if gt_positions is not None and h < len(gt_positions):
            # GT is 6 gripper node positions - compute center
            gt_center = gt_positions[h].mean(axis=0)  # [3]
            print(f"  GT Abs Position: [{gt_center[0]:.4f}, {gt_center[1]:.4f}, {gt_center[2]:.4f}]")
            
            # Compute position error
            if pred_abs_positions is not None:
                pos_error = np.linalg.norm(pred_abs_positions[h] - gt_center)
                print(f"  Position Error:  {pos_error:.4f} m")
        
        if gt_grips is not None and h < len(gt_grips):
            grip_match = (grips[h] > 0.5) == (gt_grips[h] > 0.5)
            print(f"  GT Gripper:      {'CLOSE' if gt_grips[h] > 0.5 else 'OPEN'}")
            print(f"  Gripper Match:   {'YES' if grip_match else 'NO'}")


def compute_metrics(actions: np.ndarray, grips: np.ndarray,
                   gt_positions: np.ndarray, gt_grips: np.ndarray,
                   live_pose: np.ndarray):
    """
    Compute quantitative metrics.
    
    Properly computes predicted absolute positions by cumulatively applying
    the predicted SE(3) transforms, then compares with GT absolute positions.
    
    Args:
        actions: Predicted SE(3) transforms [horizon, 4, 4] (relative transforms)
        grips: Predicted grippers [horizon, 1]
        gt_positions: Ground truth gripper positions [horizon, 6, 3]
        gt_grips: Ground truth grippers [horizon]
        live_pose: Live end-effector pose [4, 4]
    
    Returns:
        Dictionary with metrics:
        - mean_position_error: Average position error across horizon
        - max_position_error: Maximum position error
        - final_position_error: Error at the last timestep
        - gripper_accuracy: Fraction of correct gripper predictions
    """
    metrics = {}
    
    horizon = min(len(actions), len(gt_positions))
    
    # Compute predicted absolute positions by cumulative transforms
    pred_positions = compute_predicted_positions(actions, live_pose)  # [horizon, 3]
    
    # Compute GT centers (mean of 6 gripper nodes)
    gt_centers = np.array([gt_positions[h].mean(axis=0) for h in range(horizon)])  # [horizon, 3]
    
    # Position errors at each timestep
    position_errors = np.linalg.norm(pred_positions - gt_centers, axis=1)  # [horizon]
    
    metrics['mean_position_error'] = np.mean(position_errors)
    metrics['max_position_error'] = np.max(position_errors)
    metrics['final_position_error'] = position_errors[-1]
    
    # Per-axis errors (for debugging)
    axis_errors = np.abs(pred_positions - gt_centers)  # [horizon, 3]
    metrics['mean_x_error'] = np.mean(axis_errors[:, 0])
    metrics['mean_y_error'] = np.mean(axis_errors[:, 1])
    metrics['mean_z_error'] = np.mean(axis_errors[:, 2])
    
    # Gripper accuracy
    grip_correct = 0
    for h in range(min(len(grips), len(gt_grips))):
        if (grips[h] > 0.5) == (gt_grips[h] > 0.5):
            grip_correct += 1
    metrics['gripper_accuracy'] = grip_correct / min(len(grips), len(gt_grips))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Instant Policy Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--encoder_checkpoint", type=str, default="./model.pt",
                        help="Path to encoder weights (model.pt)")
    parser.add_argument("--data_dir", type=str, default="./data/pseudo_demos",
                        help="Path to pseudo-demo data")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to run inference on")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    args = parser.parse_args()
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load model
    model = load_model(args.checkpoint, args.encoder_checkpoint, args.device)
    print(f"Model loaded successfully on {args.device}")
    
    # Set inference parameters
    model.set_num_demos(2)
    model.set_num_diffusion_steps(4)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    dataset = InstantPolicyDataset(
        data_dir=args.data_dir,
        num_demos=2,
        context_len=10,  # L=10 per Appendix E
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Run inference on samples
    all_metrics = []
    
    for i in range(min(args.num_samples, len(dataset))):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i+1}/{args.num_samples}")
        print(f"{'='*60}")
        
        # Prepare sample
        full_sample, raw_sample = prepare_sample_from_dataset(dataset, i)
        
        # Get ground truth (target positions from dataset)
        gt_positions = raw_sample['target_positions'].numpy()  # [horizon, 6, 3]
        gt_grips = raw_sample['target_grips'].numpy()          # [horizon]
        live_pose = raw_sample['live_pose'].numpy()            # [4, 4]
        
        # Run inference
        print("\nRunning inference...")
        with torch.no_grad():
            try:
                actions, grips = model.predict_actions(full_sample)
                print(f"  Output shapes: actions {actions.shape}, grips {grips.shape}")
            except Exception as e:
                print(f"  Inference failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Visualize (pass live_pose for absolute position calculation)
        visualize_predictions(actions, grips, gt_positions=gt_positions, gt_grips=gt_grips, live_pose=live_pose)
        
        # Compute metrics
        metrics = compute_metrics(actions, grips, gt_positions, gt_grips, live_pose)
        all_metrics.append(metrics)
        
        print(f"\nMetrics:")
        print(f"  Mean Position Error: {metrics['mean_position_error']:.4f} m")
        print(f"  Max Position Error:  {metrics['max_position_error']:.4f} m")
        print(f"  Final Position Error: {metrics['final_position_error']:.4f} m")
        print(f"  Per-axis (X/Y/Z): {metrics['mean_x_error']:.4f} / {metrics['mean_y_error']:.4f} / {metrics['mean_z_error']:.4f} m")
        print(f"  Gripper Accuracy: {metrics['gripper_accuracy']*100:.1f}%")
    
    # Summary
    if all_metrics:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        avg_pos = np.mean([m['mean_position_error'] for m in all_metrics])
        avg_final = np.mean([m['final_position_error'] for m in all_metrics])
        avg_grip = np.mean([m['gripper_accuracy'] for m in all_metrics])
        print(f"  Average Position Error: {avg_pos:.4f} m")
        print(f"  Average Final Error:    {avg_final:.4f} m")
        print(f"  Average Gripper Accuracy: {avg_grip*100:.1f}%")


if __name__ == "__main__":
    main()
