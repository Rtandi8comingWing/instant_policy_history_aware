#!/usr/bin/env python3
"""
Script to generate ShapeNet-based demonstrations for training Instant Policy.

This script uses real 3D objects from the ShapeNet dataset instead of
procedurally generated geometric primitives.

Usage:
    # Basic usage
    python scripts/generate_shapenet_demos.py \
        --shapenet_root ./data/shapenet \
        --output_dir ./data/shapenet_demos \
        --num_tasks 10000

    # With category filtering
    python scripts/generate_shapenet_demos.py \
        --shapenet_root ./data/shapenet \
        --output_dir ./data/shapenet_demos \
        --num_tasks 10000 \
        --categories 02691156 02958343 03001627

    # Full example with all options
    python scripts/generate_shapenet_demos.py \
        --shapenet_root ./data/shapenet \
        --output_dir ./data/shapenet_demos \
        --num_tasks 10000 \
        --num_demos_per_task 2 \
        --num_objects 3 \
        --num_points 500 \
        --min_size 0.02 \
        --max_size 0.08 \
        --seed 42
"""

import argparse
import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ip_src.data.shapenet_demo import (
    ShapeNetDemoGenerator,
    ShapeNetTaskConfig,
    MotionType,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate ShapeNet-based demonstrations for Instant Policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 10000 tasks using all ShapeNet categories
  python scripts/generate_shapenet_demos.py \\
      --shapenet_root ./data/shapenet \\
      --output_dir ./data/shapenet_demos \\
      --num_tasks 10000

  # Generate tasks using only specific categories (by ID or name)
  python scripts/generate_shapenet_demos.py \\
      --shapenet_root ./data/shapenet \\
      --output_dir ./data/shapenet_demos \\
      --categories 02691156 02958343 \\
      --num_tasks 5000

ShapeNet Category IDs (common ones):
  02691156 - airplane
  02958343 - car  
  03001627 - chair
  03636649 - lamp
  04256520 - sofa
  04379243 - table
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--shapenet_root",
        type=str,
        required=True,
        help="Root directory of ShapeNet dataset containing .obj files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/shapenet_demos",
        help="Output directory for generated data (default: ./data/shapenet_demos)",
    )
    
    # Dataset size
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=10000,
        help="Number of tasks to generate (default: 10000)",
    )
    parser.add_argument(
        "--num_demos_per_task",
        type=int,
        default=2,
        help="Number of demonstrations per task (default: 2)",
    )
    
    # ShapeNet filtering
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="ShapeNet category IDs or folder names to use (default: all)",
    )
    
    # Scene configuration
    parser.add_argument(
        "--num_objects",
        type=int,
        default=3,
        help="Number of objects in each scene (default: 3)",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=500,
        help="Number of points per object point cloud (default: 500)",
    )
    parser.add_argument(
        "--min_size",
        type=float,
        default=0.02,
        help="Minimum object size in meters (default: 0.02 = 2cm)",
    )
    parser.add_argument(
        "--max_size",
        type=float,
        default=0.08,
        help="Maximum object size in meters (default: 0.08 = 8cm)",
    )
    
    # Trajectory parameters
    parser.add_argument(
        "--trans_step",
        type=float,
        default=0.01,
        help="Translation step size in meters (default: 0.01 = 1cm)",
    )
    parser.add_argument(
        "--rot_step",
        type=float,
        default=0.0524,
        help="Rotation step size in radians (default: 0.0524 = ~3 degrees)",
    )
    parser.add_argument(
        "--gripper_flip_prob",
        type=float,
        default=0.1,
        help="Probability of flipping gripper state (default: 0.1 = 10%%)",
    )
    
    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate ShapeNet path
    if not os.path.exists(args.shapenet_root):
        print(f"Error: ShapeNet root directory not found: {args.shapenet_root}")
        print("Please provide a valid path to your ShapeNet dataset.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration
    config = ShapeNetTaskConfig(
        shapenet_root=args.shapenet_root,
        allowed_categories=args.categories,
        num_objects=args.num_objects,
        num_points_per_object=args.num_points,
        object_size_range=(args.min_size, args.max_size),
        trans_step=args.trans_step,
        rot_step=args.rot_step,
        gripper_flip_prob=args.gripper_flip_prob,
    )
    
    # Print configuration
    print("=" * 60)
    print("ShapeNet Demonstration Generator")
    print("=" * 60)
    print(f"ShapeNet root:      {args.shapenet_root}")
    print(f"Output directory:   {args.output_dir}")
    print(f"Categories:         {args.categories or 'all'}")
    print(f"Number of tasks:    {args.num_tasks:,}")
    print(f"Demos per task:     {args.num_demos_per_task}")
    print(f"Objects per scene:  {args.num_objects}")
    print(f"Points per object:  {args.num_points}")
    print(f"Object size range:  {args.min_size*100:.1f}cm - {args.max_size*100:.1f}cm")
    print(f"Translation step:   {args.trans_step*100:.1f}cm")
    print(f"Rotation step:      {args.rot_step*180/3.14159:.1f} degrees")
    print(f"Gripper flip prob:  {args.gripper_flip_prob*100:.0f}%")
    print(f"Random seed:        {args.seed}")
    print("=" * 60)
    
    # Initialize generator
    print("\nInitializing ShapeNet demonstration generator...")
    try:
        generator = ShapeNetDemoGenerator(
            config=config,
            seed=args.seed,
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install trimesh: pip install trimesh")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Found {len(generator.shapenet_loader.mesh_paths):,} meshes in ShapeNet")
    
    # Generate demonstrations
    print(f"\nGenerating {args.num_tasks:,} tasks...")
    
    # Track motion type distribution
    motion_counts = {mt.value: 0 for mt in MotionType}
    
    # Setup iterator with optional progress bar
    if not args.no_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(args.num_tasks), desc="Generating tasks")
        except ImportError:
            print("Note: Install tqdm for progress bar: pip install tqdm")
            iterator = range(args.num_tasks)
    else:
        iterator = range(args.num_tasks)
    
    # Generate tasks
    for task_idx in iterator:
        # Generate task
        task_data = generator.generate_task(num_demos=args.num_demos_per_task)
        
        # Save to disk
        task_path = os.path.join(args.output_dir, f"task_{task_idx:06d}.pt")
        generator.save_task(task_data, task_path)
        
        # Track motion type
        motion_counts[task_data['motion_type']] += 1
    
    # Save metadata
    metadata = {
        'num_tasks': args.num_tasks,
        'num_demos_per_task': args.num_demos_per_task,
        'shapenet_root': args.shapenet_root,
        'allowed_categories': args.categories,
        'num_meshes_available': len(generator.shapenet_loader.mesh_paths),
        'config': {
            'num_objects': config.num_objects,
            'num_points_per_object': config.num_points_per_object,
            'object_size_range': config.object_size_range,
            'workspace_size': config.workspace_size,
            'workspace_center': config.workspace_center,
            'trans_step': config.trans_step,
            'rot_step': config.rot_step,
            'gripper_flip_prob': config.gripper_flip_prob,
            'min_waypoints': config.min_waypoints,
            'max_waypoints': config.max_waypoints,
        },
        'motion_distribution': motion_counts,
        'seed': args.seed,
    }
    torch.save(metadata, os.path.join(args.output_dir, 'metadata.pt'))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Tasks generated:  {args.num_tasks:,}")
    print(f"Total demos:      {args.num_tasks * args.num_demos_per_task:,}")
    
    print("\nMotion type distribution:")
    for motion_type, count in sorted(motion_counts.items()):
        percentage = count / args.num_tasks * 100
        print(f"  {motion_type:20s}: {count:6,} ({percentage:5.1f}%)")
    
    print("\nFiles created:")
    print(f"  - task_000000.pt to task_{args.num_tasks-1:06d}.pt")
    print(f"  - metadata.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
