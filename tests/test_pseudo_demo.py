"""
Test pseudo-demonstration generator with Appendix D improvements.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from scipy.spatial.transform import Rotation as R
from ip_src.data.pseudo_demo import (
    PseudoDemoGenerator, 
    MotionType, 
    PseudoTaskConfig,
    resample_trajectory
)


def test_resample_trajectory():
    """Test that resampling produces constant spacing."""
    print("Testing resample_trajectory...")
    
    # Create a 10cm trajectory
    positions = np.array([
        [0, 0, 0],
        [0.05, 0, 0],
        [0.1, 0, 0],
    ])
    rotations = [R.identity(), R.identity(), R.identity()]
    grippers = [1, 1, 0]
    
    resampled_pos, resampled_rot, resampled_grip = resample_trajectory(
        positions, rotations, grippers, trans_step=0.01
    )
    
    print(f"  Original length: {len(positions)}")
    print(f"  Resampled length: {len(resampled_pos)}")
    
    # Should be ~10-11 steps for 10cm with 1cm step
    assert len(resampled_pos) >= 10, f"Expected >= 10 steps, got {len(resampled_pos)}"
    
    # Check spacing is approximately constant
    diffs = np.diff(resampled_pos, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    mean_dist = np.mean(distances)
    
    print(f"  Mean step distance: {mean_dist*100:.2f}cm")
    assert 0.008 < mean_dist < 0.015, f"Step distance should be ~1cm, got {mean_dist*100:.2f}cm"
    
    print("[PASS] test_resample_trajectory")


def test_config_defaults():
    """Test that config has correct Appendix D defaults."""
    print("\nTesting PseudoTaskConfig defaults...")
    
    config = PseudoTaskConfig()
    
    # Check motion probabilities sum to 1
    total_prob = sum(config.motion_probs.values())
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities should sum to 1, got {total_prob}"
    
    # Check 50% random waypoints
    random_wp_prob = config.motion_probs.get(MotionType.RANDOM_WAYPOINTS.value, 0)
    print(f"  Random waypoints probability: {random_wp_prob*100:.0f}%")
    assert random_wp_prob == 0.5, f"Expected 50% random waypoints, got {random_wp_prob*100}%"
    
    # Check gripper flip probability
    print(f"  Gripper flip probability: {config.gripper_flip_prob*100:.0f}%")
    assert config.gripper_flip_prob == 0.1, f"Expected 10% gripper flip"
    
    # Check step sizes
    print(f"  Translation step: {config.trans_step*100:.1f}cm")
    print(f"  Rotation step: {np.degrees(config.rot_step):.1f} degrees")
    assert config.trans_step == 0.01, "Expected 1cm trans step"
    assert abs(np.degrees(config.rot_step) - 3.0) < 0.1, "Expected ~3 degree rot step"
    
    print("[PASS] test_config_defaults")


def test_generator():
    """Test PseudoDemoGenerator."""
    print("\nTesting PseudoDemoGenerator...")
    
    gen = PseudoDemoGenerator(num_objects=2, seed=42)
    
    # Generate a task
    task = gen.generate_task(num_demos=2)
    
    print(f"  Motion type: {task['motion_type']}")
    print(f"  Number of demos: {len(task['demos'])}")
    
    demo = task['demos'][0]
    traj_len = len(demo['pcds'])
    print(f"  Trajectory length: {traj_len}")
    
    # Trajectory length should be dynamic, not fixed
    assert traj_len > 0, "Trajectory should not be empty"
    
    # Check demo structure
    assert 'pcds' in demo, "Demo should have pcds"
    assert 'T_w_es' in demo, "Demo should have T_w_es"
    assert 'grips' in demo, "Demo should have grips"
    
    print("[PASS] test_generator")


def test_motion_distribution():
    """Test that motion type distribution matches Appendix D."""
    print("\nTesting motion distribution...")
    
    gen = PseudoDemoGenerator(seed=123)
    
    motion_counts = {mt.value: 0 for mt in MotionType}
    num_samples = 100
    
    for _ in range(num_samples):
        task = gen.generate_task(num_demos=1)
        motion_counts[task['motion_type']] += 1
    
    print("  Distribution:")
    for mt, count in motion_counts.items():
        print(f"    {mt}: {count}%")
    
    # Random waypoints should be ~50%
    random_wp_count = motion_counts.get(MotionType.RANDOM_WAYPOINTS.value, 0)
    assert 35 < random_wp_count < 65, \
        f"Expected ~50% random waypoints, got {random_wp_count}%"
    
    print("[PASS] test_motion_distribution")


def test_gripper_flip():
    """Test that gripper flipping is applied."""
    print("\nTesting gripper state flipping...")
    
    gen = PseudoDemoGenerator(seed=42)
    
    # Generate many demos and check for flipped states
    num_flips = 0
    total_states = 0
    
    for _ in range(10):
        task = gen.generate_task(num_demos=5)
        
        for demo in task['demos']:
            total_states += len(demo['grips'])
            # Count transitions (not perfect but gives an idea)
            for i in range(1, len(demo['grips'])):
                if demo['grips'][i] != demo['grips'][i-1]:
                    num_flips += 1
    
    flip_rate = num_flips / total_states if total_states > 0 else 0
    print(f"  Approximate flip rate: {flip_rate*100:.2f}%")
    
    # Just check that some flipping occurs
    assert num_flips > 0, "Expected some gripper state changes"
    
    print("[PASS] test_gripper_flip")


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Pseudo-Demo Generator (Appendix D)")
    print("=" * 50)
    
    test_resample_trajectory()
    test_config_defaults()
    test_generator()
    test_motion_distribution()
    test_gripper_flip()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
