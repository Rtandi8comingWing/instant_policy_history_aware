"""
Simulation Utilities for Instant Policy (ip_src version).

This module provides utilities for deploying and evaluating the Instant Policy
model in RLBench simulation environments.

Compatible with ip_src.models.GraphDiffusion.

Key functions:
- create_sim_env: Create RLBench simulation environment
- get_point_cloud: Extract point cloud from observation
- rl_bench_demo_to_sample: Convert RLBench demo to model input format
- rollout_model: Execute model rollouts and compute success rate

Usage:
    from sim_utils_ip import rollout_model
    from ip_src.models import GraphDiffusion
    
    # Load model from Lightning checkpoint (.ckpt contains complete model)
    checkpoint = torch.load('checkpoints/last.ckpt', map_location='cuda')
    hparams = checkpoint['hyper_parameters'].copy()
    hparams['device'] = 'cuda'
    hparams.pop('encoder_checkpoint', None)  # Not needed, weights are in .ckpt
    
    model = GraphDiffusion(**hparams)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    success_rate = rollout_model(model, num_demos=2, task_name='plate_out')
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm, trange

# RLBench imports
from rlbench.tasks import *
from rlbench.backend.spawn_boundary import BoundingBox
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

# ip_src imports
from ip_src.utils.sampling import sample_to_cond_demo, subsample_pcd, transform_pcd
from ip_src.utils.transforms import pose_to_transform, transform_to_pose


# ============================================================================
# RLBench Task Registry
# ============================================================================

TASK_NAMES = {
    # Manipulation tasks
    'lift_lid': TakeLidOffSaucepan,
    'phone_on_base': PhoneOnBase,
    'open_box': OpenBox,
    'slide_block': SlideBlockToTarget,
    'close_box': CloseBox,
    'basketball': BasketballInHoop,
    'buzz': BeatTheBuzz,
    'close_microwave': CloseMicrowave,
    'plate_out': TakePlateOffColoredDishRack,
    'toilet_seat_down': ToiletSeatDown,
    'toilet_seat_up': ToiletSeatUp,
    'toilet_roll_off': TakeToiletRollOffStand,
    'open_microwave': OpenMicrowave,
    'lamp_on': LampOn,
    'umbrella_out': TakeUmbrellaOutOfUmbrellaStand,
    'push_button': PushButton,
    'put_rubbish': PutRubbishInBin,
}


# ============================================================================
# Environment Utilities
# ============================================================================

def override_bounds(pos: Optional[np.ndarray], rot: float, env: Environment) -> None:
    """
    Override workspace boundary constraints for the environment.
    
    This is useful for testing or when the default boundaries are too restrictive.
    
    Args:
        pos: Optional position override [3]
        rot: Rotation bound (around z-axis)
        env: RLBench environment instance
    """
    if pos is not None:
        # Disable boundary checking
        BoundingBox.within_boundary = lambda x, y, z: True
        env._scene._workspace_boundary._boundaries[0]._get_position_within_boundary = lambda x, y: pos
    
    # Set tight rotation bounds
    env._scene.task.base_rotation_bounds = lambda: (
        (0.0, 0.0, rot - 0.0001), 
        (0.0, 0.0, rot + 0.0001)
    )


def create_sim_env(
    task_name: str,
    headless: bool = False,
    restrict_rot: bool = True,
) -> Tuple[Environment, object]:
    """
    Create and configure RLBench simulation environment.
    
    Args:
        task_name: Name of the task (must be in TASK_NAMES)
        headless: Whether to run without visualization
        restrict_rot: Whether to restrict rotation bounds
    
    Returns:
        Tuple of (environment, task)
    
    Raises:
        KeyError: If task_name is not in TASK_NAMES
    """
    if task_name not in TASK_NAMES:
        raise KeyError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(TASK_NAMES.keys())}"
        )
    
    # Configure observation
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    
    # Configure action mode: End-effector pose control + discrete gripper
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )
    
    # Create environment
    env = Environment(
        action_mode,
        './',
        obs_config=obs_config,
        headless=headless
    )
    env.launch()
    
    # Get task
    task = env.get_task(TASK_NAMES[task_name])
    
    # Override path planning to use linear paths (faster, more stable)
    def linear_path_override(
        position, 
        euler=None, 
        quaternion=None, 
        ignore_collisions=False, 
        trials=300, 
        max_configs=1,
        distance_threshold=0.65, 
        max_time_ms=10, 
        trials_per_goal=1, 
        algorithm=None, 
        relative_to=None
    ):
        return env._robot.arm.get_linear_path(
            position, euler, quaternion, 
            ignore_collisions=ignore_collisions,
            relative_to=relative_to
        )
    
    env._robot.arm.get_path = linear_path_override
    
    # Set default arm joint positions
    env._scene._start_arm_joint_pos = np.array([
        6.74760377e-05, -1.91104114e-02, -3.62065766e-05, -1.64271665e+00,
        -1.14094291e-07, 1.55336857e+00, 7.85427451e-01
    ])
    
    # Optionally restrict rotation bounds for more consistent evaluation
    if restrict_rot:
        rot_bounds = env._scene.task.base_rotation_bounds()
        mean_rot = (rot_bounds[0][2] + rot_bounds[1][2]) / 2
        env._scene.task.base_rotation_bounds = lambda: (
            (0.0, 0.0, max(rot_bounds[0][2], mean_rot - np.pi / 3)),
            (0.0, 0.0, min(rot_bounds[1][2], mean_rot + np.pi / 3))
        )
    
    return env, task


# ============================================================================
# Data Conversion Utilities
# ============================================================================

def get_point_cloud(
    obs,
    camera_names: Tuple[str, ...] = ('front', 'left_shoulder', 'right_shoulder'),
    mask_threshold: int = 60,
) -> np.ndarray:
    """
    Extract and merge point clouds from multiple camera views.
    
    Uses segmentation masks to filter out background/robot points.
    
    Args:
        obs: RLBench observation object
        camera_names: Camera names to use
        mask_threshold: Threshold for segmentation mask (higher = more selective)
    
    Returns:
        Merged point cloud [N, 3] in world frame
    """
    pcds = []
    
    for camera_name in camera_names:
        # Get ordered point cloud from camera
        ordered_pcd = getattr(obs, f'{camera_name}_point_cloud')
        
        # Get segmentation mask
        mask = getattr(obs, f'{camera_name}_mask')
        
        # Filter points using mask (keep object points, filter robot/background)
        # Higher mask values typically correspond to task-relevant objects
        masked_pcd = ordered_pcd[mask > mask_threshold]
        
        pcds.append(masked_pcd)
    
    # Concatenate all camera views
    if len(pcds) > 0 and any(len(p) > 0 for p in pcds):
        return np.concatenate([p for p in pcds if len(p) > 0], axis=0)
    else:
        # Return empty point cloud if no valid points
        return np.zeros((0, 3))


def rl_bench_demo_to_sample(demo: List) -> Dict:
    """
    Convert RLBench demonstration to raw sample format.
    
    This converts the demonstration into the format expected by sample_to_cond_demo.
    
    Args:
        demo: List of RLBench observations from a demonstration
    
    Returns:
        Dictionary with:
            - 'pcds': List of point clouds [T][N, 3] in world frame
            - 'T_w_es': List of end-effector transforms [T][4, 4]
            - 'grips': List of gripper states [T] (0=closed, 1=open)
    """
    sample = {
        'pcds': [],
        'T_w_es': [],
        'grips': [],
    }
    
    for obs in demo:
        # Extract point cloud
        pcd = get_point_cloud(obs)
        sample['pcds'].append(pcd)
        
        # Convert gripper pose to 4x4 transform matrix
        # obs.gripper_pose is [x, y, z, qx, qy, qz, qw]
        T_w_e = pose_to_transform(obs.gripper_pose)
        sample['T_w_es'].append(T_w_e)
        
        # Gripper state (1 = open, 0 = closed)
        sample['grips'].append(obs.gripper_open)
    
    return sample


def prepare_live_observation(
    obs,
    num_points: int = 2048,
) -> Dict:
    """
    Prepare current observation for model inference.
    
    Converts the live observation to the format expected by the model's
    predict_actions method.
    
    Args:
        obs: RLBench observation object
        num_points: Number of points to subsample
    
    Returns:
        Dictionary with:
            - 'obs': List containing single point cloud in EE frame [1][num_points, 3]
            - 'T_w_es': List containing single transform [1][4, 4]
            - 'grips': List containing single gripper state [1]
    """
    # Get end-effector transform
    T_w_e = pose_to_transform(obs.gripper_pose)
    T_e_w = np.linalg.inv(T_w_e)
    
    # Get point cloud, subsample, and transform to EE frame
    pcd_world = get_point_cloud(obs)
    pcd_subsampled = subsample_pcd(pcd_world, num_points=num_points)
    pcd_ee = transform_pcd(pcd_subsampled, T_e_w)
    
    return {
        'obs': [pcd_ee],
        'T_w_es': [T_w_e],
        'grips': [obs.gripper_open],
    }


# ============================================================================
# Evaluation Functions
# ============================================================================

def execute_action(
    task,
    T_w_e: np.ndarray,
    action: np.ndarray,
    gripper: float,
) -> Tuple[object, float, bool]:
    """
    Execute a single action in the environment.
    
    Computes the target world pose from relative action and executes it.
    
    Args:
        task: RLBench task object
        T_w_e: Current end-effector pose in world frame [4, 4]
        action: Relative action transform T_EA [4, 4]
        gripper: Gripper command (0-1, thresholded at 0.5)
    
    Returns:
        Tuple of (observation, reward, terminate)
    """
    # Compute target world pose: T_WA = T_WE @ T_EA
    T_w_target = T_w_e @ action
    
    # Convert to pose format [x, y, z, qx, qy, qz, qw]
    target_pose = transform_to_pose(T_w_target)
    
    # Build action array [7 pose + 1 gripper]
    env_action = np.zeros(8)
    env_action[:7] = target_pose
    
    # Threshold gripper command (original uses: int((grips[j] + 1) / 2 > 0.5))
    # Our model outputs 0-1, so we just threshold at 0.5
    env_action[7] = int(gripper > 0.5)
    
    # Execute action
    return task.step(env_action)


def rollout_model(
    model,
    num_demos: int,
    task_name: str = 'phone_on_base',
    num_rollouts: int = 10,
    max_execution_steps: int = 30,
    execution_horizon: int = 8,
    num_traj_wp: int = 10,
    num_points: int = 2048,
    headless: bool = True,
    restrict_rot: bool = True,
    verbose: bool = True,
) -> float:
    """
    Execute model rollouts in simulation and compute success rate.
    
    This is the main evaluation function that:
    1. Collects demonstration data from the environment
    2. Runs multiple rollouts using the model
    3. Computes and returns the success rate
    
    Args:
        model: GraphDiffusion model (must have predict_actions method)
        num_demos: Number of demonstrations to collect for context
        task_name: Name of the RLBench task
        num_rollouts: Number of evaluation rollouts
        max_execution_steps: Maximum action prediction steps per rollout
        execution_horizon: Number of action steps to execute per prediction
        num_traj_wp: Number of waypoints per demonstration context
        num_points: Number of points to subsample per point cloud
        headless: Whether to run without visualization
        restrict_rot: Whether to restrict rotation bounds
        verbose: Whether to show progress bars
    
    Returns:
        Success rate (0.0 to 1.0)
    """
    # ========================================================================
    # Create simulation environment
    # ========================================================================
    env, task = create_sim_env(task_name, headless=headless, restrict_rot=restrict_rot)
    
    # ========================================================================
    # Collect demonstrations
    # ========================================================================
    full_sample = {
        'demos': [dict()] * num_demos,
        'live': dict(),
    }
    
    demo_iterator = tqdm(range(num_demos), desc='Collecting demos', leave=False) if verbose else range(num_demos)
    
    for i in demo_iterator:
        done = False
        while not done:
            try:
                # Get demonstration from environment
                demos = task.get_demos(1, live_demos=True, max_attempts=1000)
                
                # Convert to sample format
                sample = rl_bench_demo_to_sample(demos[0])
                
                # Process for conditioning (subsample waypoints, transform to EE frame)
                cond_demo = sample_to_cond_demo(sample, num_traj_wp, num_points=num_points)
                
                # Verify correct number of waypoints
                assert len(cond_demo['obs']) == num_traj_wp, \
                    f"Expected {num_traj_wp} waypoints, got {len(cond_demo['obs'])}"
                
                full_sample['demos'][i] = cond_demo
                done = True
                
            except Exception as e:
                # Retry on failure
                if verbose:
                    print(f"Demo collection failed: {e}, retrying...")
                continue
    
    # ========================================================================
    # Run evaluation rollouts
    # ========================================================================
    successes = []
    
    if verbose:
        pbar = trange(num_rollouts, desc=f'Evaluating model, SR: 0/{num_rollouts}', leave=False)
    else:
        pbar = range(num_rollouts)
    
    for rollout_idx in pbar:
        # Reset environment
        done = False
        while not done:
            try:
                task.reset()
                done = True
            except Exception as e:
                if verbose:
                    print(f"Reset failed: {e}, retrying...")
                continue
        
        # Run rollout
        success = 0
        
        for step in range(max_execution_steps):
            # Get current observation
            curr_obs = task.get_observation()
            
            # Prepare live observation for model
            full_sample['live'] = prepare_live_observation(curr_obs, num_points=num_points)
            
            # Get current end-effector pose for action computation
            T_w_e = pose_to_transform(curr_obs.gripper_pose)
            
            # Predict actions using model
            # Returns: actions [horizon, 4, 4], grips [horizon, 1]
            actions, grips = model.predict_actions(full_sample)
            
            # Execute predicted actions
            terminate = False
            
            for j in range(execution_horizon):
                try:
                    # Execute single action
                    # actions[j] is T_EA (relative transform in EE frame)
                    curr_obs, reward, terminate = execute_action(
                        task, 
                        T_w_e, 
                        actions[j], 
                        grips[j, 0] if grips.ndim > 1 else grips[j]
                    )
                    
                    # Check for success
                    success = int(terminate and reward > 0.)
                    
                    # Update T_w_e for next action in sequence
                    T_w_e = pose_to_transform(curr_obs.gripper_pose)
                    
                except Exception as e:
                    if verbose:
                        print(f"Action execution failed: {e}")
                    terminate = True
                
                if terminate:
                    break
            
            if terminate:
                break
        
        # Record result
        successes.append(success)
        
        if verbose:
            pbar.set_description(f'Evaluating model, SR: {sum(successes)}/{len(successes)}')
            pbar.refresh()
    
    if verbose:
        pbar.close()
    
    # Shutdown environment
    env.shutdown()
    
    # Compute and return success rate
    success_rate = sum(successes) / len(successes) if successes else 0.0
    
    return success_rate


def batch_evaluate_tasks(
    model,
    task_names: List[str],
    num_demos: int = 2,
    num_rollouts: int = 10,
    **kwargs,
) -> Dict[str, float]:
    """
    Evaluate model on multiple tasks.
    
    Args:
        model: GraphDiffusion model
        task_names: List of task names to evaluate
        num_demos: Number of demonstrations per task
        num_rollouts: Number of rollouts per task
        **kwargs: Additional arguments to pass to rollout_model
    
    Returns:
        Dictionary mapping task names to success rates
    """
    results = {}
    
    for task_name in task_names:
        print(f"\nEvaluating task: {task_name}")
        try:
            sr = rollout_model(
                model, 
                num_demos, 
                task_name, 
                num_rollouts=num_rollouts,
                **kwargs
            )
            results[task_name] = sr
            print(f"  Success rate: {sr:.2%}")
        except Exception as e:
            print(f"  Failed: {e}")
            results[task_name] = None
    
    return results
