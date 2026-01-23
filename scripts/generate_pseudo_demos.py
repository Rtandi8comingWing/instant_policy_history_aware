#!/usr/bin/env python3
"""
Script to generate pseudo-demonstrations for training Instant Policy.

Usage:
    python scripts/generate_pseudo_demos.py --output_dir ./data/pseudo_demos --num_tasks 10000
"""

import argparse
import os
from tqdm import tqdm

from ip_src.data.pseudo_demo import PseudoDemoGenerator


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
        "--trajectory_length",
        type=int,
        default=20,
        help="Length of each trajectory",
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
        trajectory_length=args.trajectory_length,
        seed=args.seed,
    )
    
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
    
    print(f"Generated {args.num_tasks} pseudo-tasks")
    print(f"Saved to: {args.output_dir}")
    
    # Print statistics
    total_demos = args.num_tasks * args.num_demos_per_task
    total_frames = total_demos * args.trajectory_length
    print(f"\nStatistics:")
    print(f"  Total demonstrations: {total_demos:,}")
    print(f"  Total frames: {total_frames:,}")


if __name__ == "__main__":
    main()
