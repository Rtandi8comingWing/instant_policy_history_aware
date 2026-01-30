#!/usr/bin/env python3
"""
Script to generate pseudo-demonstrations for training Instant Policy.

Usage:
    python scripts/generate_pseudo_demos.py --output_dir ./data/pseudo_demos --num_tasks 10000

Note: Trajectory length is now dynamically determined by resample_trajectory
      (1cm translation step, 3 degrees rotation step) per Appendix D.
"""

import argparse
import os
import torch
from tqdm import tqdm

from ip_src.data.pseudo_demo import PseudoDemoGenerator, MotionType


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-demonstrations")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/pseudo_demos",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=10000,
        help="Number of pseudo-tasks to generate",
    )
    parser.add_argument(
        "--num_demos_per_task",
        type=int,
        default=2,
        help="Number of demonstrations per task",
    )
    parser.add_argument(
        "--num_objects",
        type=int,
        default=3,
        help="Number of objects in each scene",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create generator
    print("Initializing pseudo-demonstration generator...")
    generator = PseudoDemoGenerator(
        num_objects=args.num_objects,
        seed=args.seed,
    )
    
    # Track motion type distribution
    motion_counts = {mt.value: 0 for mt in MotionType}
    
    # Generate pseudo-demonstrations
    print(f"Generating {args.num_tasks} pseudo-tasks...")
    for task_idx in tqdm(range(args.num_tasks)):
        # Generate demonstrations for this task
        task_data = generator.generate_task(
            num_demos=args.num_demos_per_task,
        )
        
        # Save to disk
        task_path = os.path.join(args.output_dir, f"task_{task_idx:06d}.pt")
        generator.save_task(task_data, task_path)
        
        # Track motion type
        motion_counts[task_data['motion_type']] += 1
    
    # Save metadata
    metadata = {
        'num_tasks': args.num_tasks,
        'num_demos_per_task': args.num_demos_per_task,
        'num_objects': args.num_objects,
        'motion_distribution': motion_counts,
        'seed': args.seed,
    }
    torch.save(metadata, os.path.join(args.output_dir, 'metadata.pt'))
    
    print(f"\nGenerated {args.num_tasks} pseudo-tasks")
    print(f"Saved to: {args.output_dir}")
    
    # Print statistics
    total_demos = args.num_tasks * args.num_demos_per_task
    print(f"\nStatistics:")
    print(f"  Total demonstrations: {total_demos:,}")
    print(f"  Trajectory length: Dynamic (1cm/3deg steps per Appendix D)")
    
    print("\nMotion type distribution:")
    for motion_type, count in sorted(motion_counts.items()):
        percentage = count / args.num_tasks * 100
        print(f"  {motion_type:20s}: {count:6,} ({percentage:5.1f}%)")


if __name__ == "__main__":
    main()
