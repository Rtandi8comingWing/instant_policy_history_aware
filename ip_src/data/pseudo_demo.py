"""
Pseudo-Demonstration Generator for Instant Policy.

Implements the pseudo-demonstration generation from Section 3.4 and Appendix D:
- Procedurally generates semantically consistent trajectory sets
- No real task semantics needed, just motion consistency
- Virtually infinite training data through simulation

Key features (Appendix D):
1. Object attachment: Objects follow gripper during grasping
2. Motion primitives: Pick-and-place, push, reach-and-grasp (50%)
3. Random waypoints: Fully random trajectories for generalization (50%)
4. Constant step spacing: 1cm translation and 3 degrees rotation
5. Gripper state flipping: 10% random state changes for robustness

Key insight from the paper:
"In traditional BC, the model's weights directly encode policies for a specific
set of tasks, whereas in ICIL, the model's weights should encode a more general,
task-agnostic ability to interpret and act upon the given context."

Paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion" (ICLR 2025)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import CubicSpline, interp1d
from enum import Enum


class MotionType(Enum):
    """Types of motion primitives."""
    REACH_AND_GRASP = "reach_and_grasp"
    PICK_AND_PLACE = "pick_and_place"
    PUSH = "push"
    RANDOM_WAYPOINTS = "random_waypoints"  # Renamed from RANDOM_SPLINE per Appendix D


@dataclass
class PseudoTaskConfig:
    """
    Configuration for pseudo-task generation (Appendix D).
    
    Key parameters from paper:
    - trans_step: 1cm (0.01m) constant spacing between steps
    - rot_step: 3 degrees (~0.0524 rad) constant rotation spacing
    - gripper_flip_prob: 10% probability of flipping gripper state
    - 50% random waypoints, 50% biased sampling (pick, push, grasp)
    """
    num_objects: int = 3
    # Removed: trajectory_length - now determined dynamically by resampling
    workspace_size: Tuple[float, float, float] = (0.5, 0.5, 0.3)
    workspace_center: Tuple[float, float, float] = (0.5, 0.0, 0.2)
    object_size_range: Tuple[float, float] = (0.02, 0.08)
    num_points_per_object: int = 500
    noise_scale: float = 0.01
    attachment_threshold: float = 0.05  # Distance threshold for object attachment
    
    # Resampling parameters (Appendix D: "1cm and 3 degrees")
    trans_step: float = 0.01  # 1cm translation step
    rot_step: float = 0.0524  # ~3 degrees in radians
    
    # Data augmentation (Appendix D: "10% gripper flip")
    gripper_flip_prob: float = 0.1
    
    # Random waypoints parameters (Appendix D: "2 to 6 waypoints")
    min_waypoints: int = 2
    max_waypoints: int = 6
    
    # Motion primitive probabilities (Appendix D: 50% random, 50% biased)
    motion_probs: Dict[str, float] = None
    
    def __post_init__(self):
        if self.motion_probs is None:
            # Appendix D: "half of the samples use bias sampling... 
            # the rest use completely random waypoints"
            self.motion_probs = {
                MotionType.RANDOM_WAYPOINTS.value: 0.50,  # 50% random waypoints
                MotionType.PICK_AND_PLACE.value: 0.20,    # Biased sampling
                MotionType.REACH_AND_GRASP.value: 0.15,   # Biased sampling
                MotionType.PUSH.value: 0.15,              # Biased sampling
            }


class ObjectState:
    """
    Tracks the state of an object during trajectory generation.
    
    Handles attachment logic: when gripper is closed and near object,
    the object follows the gripper's rigid body motion.
    """
    
    def __init__(
        self,
        position: np.ndarray,
        size: float,
        shape: str,
        pcd: np.ndarray,
    ):
        self.initial_position = position.copy()
        self.position = position.copy()
        self.size = size
        self.shape = shape
        self.base_pcd = pcd.copy()  # Local coordinates centered at origin
        self.current_pcd = pcd.copy()
        self.attached = False
        self.attachment_offset = None  # Offset from gripper when attached
        
        # Store local coordinates (centered at object position)
        self.local_pcd = pcd - position
    
    def update(
        self,
        gripper_position: np.ndarray,
        gripper_rotation: np.ndarray,
        gripper_closed: bool,
        attachment_threshold: float,
    ):
        """
        Update object state based on gripper interaction.
        
        Args:
            gripper_position: Current gripper position [3]
            gripper_rotation: Current gripper rotation matrix [3, 3]
            gripper_closed: Whether gripper is closed
            attachment_threshold: Distance threshold for attachment
        """
        dist = np.linalg.norm(gripper_position - self.position)
        
        if gripper_closed and dist < attachment_threshold:
            if not self.attached:
                # Start attachment: record offset in gripper frame
                self.attached = True
                # Compute offset in gripper local frame
                world_offset = self.position - gripper_position
                self.attachment_offset = gripper_rotation.T @ world_offset
                self.attachment_rotation = gripper_rotation.T  # Initial rotation offset
            
            # Apply gripper transform to object
            # Object position follows gripper
            self.position = gripper_position + gripper_rotation @ self.attachment_offset
            
            # Transform point cloud: apply gripper rotation to local coordinates
            self.current_pcd = (gripper_rotation @ self.local_pcd.T).T + self.position
        else:
            if self.attached and not gripper_closed:
                # Detach: object stays at current position
                self.attached = False
                # Update local_pcd to current orientation
                self.local_pcd = self.current_pcd - self.position
            elif not self.attached:
                # Object stationary
                self.current_pcd = self.local_pcd + self.position
    
    def get_pcd(self) -> np.ndarray:
        """Get current point cloud."""
        return self.current_pcd.copy()
    
    def reset(self):
        """Reset object to initial state."""
        self.position = self.initial_position.copy()
        self.current_pcd = self.base_pcd.copy()
        self.local_pcd = self.base_pcd - self.initial_position
        self.attached = False
        self.attachment_offset = None


def resample_trajectory(
    positions: np.ndarray,
    rotations: List[R],
    grippers: List[int],
    trans_step: float = 0.01,
    rot_step: float = 0.0524,
) -> Tuple[np.ndarray, List[R], List[int]]:
    """
    Resample trajectory to ensure constant spacing (Appendix D).
    
    "We ensure that the spacing between subsequent steps is constant and 
    uniform (1cm and 3 degrees)."
    
    Uses the larger of translation or rotation distance to determine step count,
    then interpolates both position (linear) and rotation (Slerp).
    
    Args:
        positions: Original positions [N, 3]
        rotations: Original rotations (list of scipy Rotation objects)
        grippers: Original gripper states (list of 0/1)
        trans_step: Translation step size in meters (default: 1cm)
        rot_step: Rotation step size in radians (default: ~3 degrees)
    
    Returns:
        resampled_positions: [M, 3] with constant spacing
        resampled_rotations: List of M Rotation objects
        resampled_grippers: List of M gripper states
    """
    if len(positions) < 2:
        return positions, rotations, grippers
    
    # Compute cumulative arc length (translation)
    diffs = np.diff(positions, axis=0)
    trans_distances = np.linalg.norm(diffs, axis=1)
    cumulative_trans = np.concatenate([[0], np.cumsum(trans_distances)])
    total_trans = cumulative_trans[-1]
    
    # Compute cumulative rotation distance
    rot_distances = []
    for i in range(len(rotations) - 1):
        # Rotation distance as angle magnitude
        relative_rot = rotations[i].inv() * rotations[i + 1]
        angle = np.linalg.norm(relative_rot.as_rotvec())
        rot_distances.append(angle)
    cumulative_rot = np.concatenate([[0], np.cumsum(rot_distances)])
    total_rot = cumulative_rot[-1]
    
    # Determine number of resampled points based on larger distance
    num_trans_steps = max(1, int(np.ceil(total_trans / trans_step)))
    num_rot_steps = max(1, int(np.ceil(total_rot / rot_step)))
    num_steps = max(num_trans_steps, num_rot_steps, 2)  # At least 2 points
    
    # Create uniform parameter values for resampling
    t_uniform = np.linspace(0, 1, num_steps)
    
    # Normalize cumulative distances to [0, 1]
    if total_trans > 1e-8:
        t_trans = cumulative_trans / total_trans
    else:
        t_trans = np.linspace(0, 1, len(positions))
    
    if total_rot > 1e-8:
        t_rot = cumulative_rot / total_rot
    else:
        t_rot = np.linspace(0, 1, len(rotations))
    
    # Interpolate positions (linear interpolation along trajectory)
    resampled_positions = np.zeros((num_steps, 3))
    for dim in range(3):
        interp_func = interp1d(t_trans, positions[:, dim], kind='linear', 
                               fill_value='extrapolate')
        resampled_positions[:, dim] = interp_func(t_uniform)
    
    # Interpolate rotations using Slerp
    # Need to handle edge cases for Slerp
    if len(rotations) >= 2:
        # Ensure t_rot is strictly increasing for Slerp
        # Add small epsilon to avoid duplicate values
        t_rot_unique = t_rot.copy()
        for i in range(1, len(t_rot_unique)):
            if t_rot_unique[i] <= t_rot_unique[i-1]:
                t_rot_unique[i] = t_rot_unique[i-1] + 1e-8
        
        # Create rotation stack
        rot_stack = R.from_quat([r.as_quat() for r in rotations])
        
        try:
            slerp = Slerp(t_rot_unique, rot_stack)
            resampled_rotations = [slerp(t) for t in t_uniform]
        except ValueError:
            # Fallback: nearest neighbor interpolation
            resampled_rotations = []
            for t in t_uniform:
                idx = np.argmin(np.abs(t_rot - t))
                resampled_rotations.append(rotations[idx])
    else:
        resampled_rotations = [rotations[0]] * num_steps
    
    # Interpolate gripper states (nearest neighbor - discrete values)
    t_grip = np.linspace(0, 1, len(grippers))
    resampled_grippers = []
    for t in t_uniform:
        idx = np.argmin(np.abs(t_grip - t))
        resampled_grippers.append(grippers[idx])
    
    return resampled_positions, resampled_rotations, resampled_grippers


class MotionPrimitive:
    """
    Motion primitives for generating diverse pseudo-demonstrations.
    
    Different motion patterns to create varied training data.
    """
    
    @staticmethod
    def reach_and_grasp(
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        rng: np.random.RandomState,
        num_keypoints: int = 10,
    ) -> Tuple[np.ndarray, List[R], List[int]]:
        """
        Generate reach-and-grasp motion.
        
        Motion: Approach target -> Grasp
        
        Args:
            start_pos: Starting position [3]
            target_pos: Target position [3]
            rng: Random number generator
            num_keypoints: Number of keypoints (will be resampled later)
        
        Returns:
            positions: [num_keypoints, 3]
            rotations: List of Rotation objects
            grippers: List of gripper states (1=open, 0=closed)
        """
        positions = np.zeros((num_keypoints, 3))
        
        # Approach phase (70% of keypoints)
        approach_len = int(num_keypoints * 0.7)
        for i in range(approach_len):
            t = i / max(approach_len - 1, 1)
            positions[i] = start_pos + t * (target_pos - start_pos)
            # Add slight arc for more natural motion
            positions[i, 2] += 0.05 * np.sin(t * np.pi)
        
        # Grasp phase (30% of keypoints) - stay near target
        for i in range(approach_len, num_keypoints):
            positions[i] = target_pos + rng.randn(3) * 0.002
        
        # Orientations - gripper pointing down
        base_rot = R.from_euler('xyz', [0, np.pi, 0])
        rotations = []
        for i in range(num_keypoints):
            # Add small rotation noise
            noise_rot = R.from_rotvec(rng.randn(3) * 0.02)
            rotations.append(noise_rot * base_rot)
        
        # Grippers: open during approach, closed during grasp
        grippers = [1] * approach_len + [0] * (num_keypoints - approach_len)
        
        return positions, rotations, grippers
    
    @staticmethod
    def pick_and_place(
        start_pos: np.ndarray,
        pick_pos: np.ndarray,
        place_pos: np.ndarray,
        rng: np.random.RandomState,
        num_keypoints: int = 30,
    ) -> Tuple[np.ndarray, List[R], List[int]]:
        """
        Generate pick-and-place motion.
        
        Motion: Approach -> Grasp -> Lift -> Move -> Lower -> Release
        
        Args:
            start_pos: Starting position [3]
            pick_pos: Pick position [3]
            place_pos: Place position [3]
            rng: Random number generator
            num_keypoints: Number of keypoints (will be resampled later)
        
        Returns:
            positions: [num_keypoints, 3]
            rotations: List of Rotation objects
            grippers: List of gripper states
        """
        positions = np.zeros((num_keypoints, 3))
        
        # Phase lengths
        approach_len = int(num_keypoints * 0.2)
        grasp_len = int(num_keypoints * 0.1)
        lift_len = int(num_keypoints * 0.2)
        move_len = int(num_keypoints * 0.3)
        lower_len = int(num_keypoints * 0.1)
        release_len = num_keypoints - approach_len - grasp_len - lift_len - move_len - lower_len
        
        lift_height = 0.15  # How high to lift
        
        idx = 0
        
        # 1. Approach to pick position
        for i in range(approach_len):
            t = i / max(approach_len - 1, 1)
            positions[idx] = start_pos + t * (pick_pos - start_pos)
            idx += 1
        
        # 2. Grasp (stay at pick position)
        for i in range(grasp_len):
            positions[idx] = pick_pos + rng.randn(3) * 0.001
            idx += 1
        
        # 3. Lift
        lift_pos = pick_pos.copy()
        lift_pos[2] += lift_height
        for i in range(lift_len):
            t = i / max(lift_len - 1, 1)
            positions[idx] = pick_pos + t * (lift_pos - pick_pos)
            idx += 1
        
        # 4. Move to above place position
        above_place = place_pos.copy()
        above_place[2] += lift_height
        for i in range(move_len):
            t = i / max(move_len - 1, 1)
            positions[idx] = lift_pos + t * (above_place - lift_pos)
            idx += 1
        
        # 5. Lower
        for i in range(lower_len):
            t = i / max(lower_len - 1, 1)
            positions[idx] = above_place + t * (place_pos - above_place)
            idx += 1
        
        # 6. Release (stay at place position)
        for i in range(release_len):
            positions[idx] = place_pos + rng.randn(3) * 0.001
            idx += 1
        
        # Pad if necessary
        while idx < num_keypoints:
            positions[idx] = place_pos
            idx += 1
        
        # Orientations - gripper pointing down with slight variations
        base_rot = R.from_euler('xyz', [0, np.pi, 0])
        rotations = []
        for i in range(num_keypoints):
            noise_rot = R.from_rotvec(rng.randn(3) * 0.02)
            rotations.append(noise_rot * base_rot)
        
        # Grippers
        grippers = (
            [1] * approach_len +          # Open during approach
            [0] * (grasp_len + lift_len + move_len + lower_len) +  # Closed while holding
            [1] * release_len             # Open to release
        )
        # Ensure correct length
        while len(grippers) < num_keypoints:
            grippers.append(1)
        grippers = grippers[:num_keypoints]
        
        return positions, rotations, grippers
    
    @staticmethod
    def push(
        start_pos: np.ndarray,
        push_start: np.ndarray,
        push_end: np.ndarray,
        rng: np.random.RandomState,
        num_keypoints: int = 20,
    ) -> Tuple[np.ndarray, List[R], List[int]]:
        """
        Generate pushing motion.
        
        Motion: Approach -> Lower -> Push -> Lift
        
        Args:
            start_pos: Starting position [3]
            push_start: Where to start pushing [3]
            push_end: Where to end pushing [3]
            rng: Random number generator
            num_keypoints: Number of keypoints (will be resampled later)
        
        Returns:
            positions: [num_keypoints, 3]
            rotations: List of Rotation objects
            grippers: List of gripper states
        """
        positions = np.zeros((num_keypoints, 3))
        
        # Phase lengths
        approach_len = int(num_keypoints * 0.25)
        lower_len = int(num_keypoints * 0.1)
        push_len = int(num_keypoints * 0.5)
        lift_len = num_keypoints - approach_len - lower_len - push_len
        
        # Push approach point (above push start)
        approach_pos = push_start.copy()
        approach_pos[2] += 0.05
        
        idx = 0
        
        # 1. Approach to above push start
        for i in range(approach_len):
            t = i / max(approach_len - 1, 1)
            positions[idx] = start_pos + t * (approach_pos - start_pos)
            idx += 1
        
        # 2. Lower to push start
        for i in range(lower_len):
            t = i / max(lower_len - 1, 1)
            positions[idx] = approach_pos + t * (push_start - approach_pos)
            idx += 1
        
        # 3. Push
        for i in range(push_len):
            t = i / max(push_len - 1, 1)
            positions[idx] = push_start + t * (push_end - push_start)
            idx += 1
        
        # 4. Lift
        lift_pos = push_end.copy()
        lift_pos[2] += 0.1
        for i in range(lift_len):
            t = i / max(lift_len - 1, 1)
            positions[idx] = push_end + t * (lift_pos - push_end)
            idx += 1
        
        # Pad if necessary
        while idx < num_keypoints:
            positions[idx] = lift_pos
            idx += 1
        
        # Orientations (tilted forward for pushing)
        push_dir = push_end - push_start
        push_dir[2] = 0  # Project to horizontal plane
        push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)
        
        rotations = []
        base_rot = R.from_euler('xyz', [0.3, np.pi, 0])  # Tilted forward
        for i in range(num_keypoints):
            noise_rot = R.from_rotvec(rng.randn(3) * 0.02)
            rotations.append(noise_rot * base_rot)
        
        # Grippers (closed for pushing)
        grippers = [0] * num_keypoints
        
        return positions, rotations, grippers
    
    @staticmethod
    def random_waypoints(
        workspace_center: np.ndarray,
        workspace_size: np.ndarray,
        rng: np.random.RandomState,
        min_waypoints: int = 2,
        max_waypoints: int = 6,
        initial_samples: int = 50,  # Initial dense sampling before resampling
    ) -> Tuple[np.ndarray, List[R], List[int]]:
        """
        Generate random waypoint trajectory (Appendix D).
        
        "Random waypoints: randomly select 2 to 6 waypoints and interpolate"
        
        This generates truly random trajectories within the workspace for
        general skill acquisition. Unlike biased primitives, this has no
        semantic structure.
        
        Args:
            workspace_center: Center of workspace [3]
            workspace_size: Size of workspace [3]
            rng: Random number generator
            min_waypoints: Minimum number of waypoints (default: 2)
            max_waypoints: Maximum number of waypoints (default: 6)
            initial_samples: Initial dense sampling for cubic spline
        
        Returns:
            positions: [N, 3] positions (to be resampled later)
            rotations: List of N Rotation objects
            grippers: List of N gripper states
        """
        # Random number of waypoints (2 to 6)
        num_waypoints = rng.randint(min_waypoints, max_waypoints + 1)
        
        # Generate random waypoints within workspace
        keypoints = np.zeros((num_waypoints, 3))
        for i in range(num_waypoints):
            keypoints[i] = workspace_center + (rng.random(3) - 0.5) * workspace_size
        
        # Generate random orientations at each waypoint
        key_rotations = []
        base_rot = R.from_euler('xyz', [0, np.pi, 0])  # Gripper pointing down base
        for i in range(num_waypoints):
            # Random rotation variation
            random_euler = rng.uniform(-0.5, 0.5, 3)  # +/- ~30 degrees
            random_euler[1] = np.pi + random_euler[1] * 0.3  # Keep mostly pointing down
            key_rotations.append(R.from_euler('xyz', random_euler))
        
        # Interpolate positions using cubic spline
        t_keypoints = np.linspace(0, 1, num_waypoints)
        t_full = np.linspace(0, 1, initial_samples)
        
        positions = np.zeros((initial_samples, 3))
        for dim in range(3):
            if num_waypoints >= 4:
                cs = CubicSpline(t_keypoints, keypoints[:, dim])
            else:
                # Use linear for fewer points
                cs = interp1d(t_keypoints, keypoints[:, dim], kind='linear',
                             fill_value='extrapolate')
            positions[:, dim] = cs(t_full)
        
        # Interpolate rotations using Slerp
        if num_waypoints >= 2:
            rot_stack = R.from_quat([r.as_quat() for r in key_rotations])
            slerp = Slerp(t_keypoints, rot_stack)
            rotations = [slerp(t) for t in t_full]
        else:
            rotations = [key_rotations[0]] * initial_samples
        
        # Random gripper state changes at random waypoints
        # Appendix D: "randomly assign gripper state changes at waypoints"
        grippers = [1] * initial_samples  # Start open
        
        # Randomly select 1-2 waypoints where gripper state changes
        num_changes = rng.randint(1, 3)
        change_waypoints = rng.choice(num_waypoints, min(num_changes, num_waypoints), 
                                       replace=False)
        
        current_state = 1  # Start open
        for i, t in enumerate(t_full):
            # Check if we passed a change waypoint
            for wp_idx in change_waypoints:
                wp_t = t_keypoints[wp_idx]
                if t >= wp_t and i > 0:
                    # Find if this is the first time we pass this waypoint
                    prev_t = t_full[i - 1]
                    if prev_t < wp_t <= t:
                        current_state = 1 - current_state  # Toggle
            grippers[i] = current_state
        
        return positions, rotations, grippers
    
    @staticmethod
    def random_spline(
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        rng: np.random.RandomState,
        num_keypoints: int = 5,
        initial_samples: int = 50,
    ) -> Tuple[np.ndarray, List[R], List[int]]:
        """
        Generate random spline trajectory (legacy fallback).
        
        Kept for backward compatibility. Prefer random_waypoints.
        
        Args:
            start_pos: Starting position [3]
            end_pos: Ending position [3]
            rng: Random number generator
            num_keypoints: Number of spline keypoints
            initial_samples: Initial dense sampling
        
        Returns:
            positions: [N, 3]
            rotations: List of Rotation objects
            grippers: List of gripper states
        """
        # Generate keypoints
        keypoints = np.zeros((num_keypoints, 3))
        keypoints[0] = start_pos
        keypoints[-1] = end_pos
        
        for i in range(1, num_keypoints - 1):
            t = i / (num_keypoints - 1)
            keypoints[i] = (1 - t) * start_pos + t * end_pos
            keypoints[i] += (rng.random(3) - 0.5) * 0.15
        
        # Interpolate using cubic spline
        t_keypoints = np.linspace(0, 1, num_keypoints)
        t_full = np.linspace(0, 1, initial_samples)
        
        positions = np.zeros((initial_samples, 3))
        for dim in range(3):
            cs = CubicSpline(t_keypoints, keypoints[:, dim])
            positions[:, dim] = cs(t_full)
        
        # Orientations
        base_rot = R.from_euler('xyz', [0, np.pi, 0])
        rotations = [base_rot] * initial_samples
        
        # Grippers: open at start, close at 70%
        close_point = int(initial_samples * 0.7)
        grippers = [1] * close_point + [0] * (initial_samples - close_point)
        
        return positions, rotations, grippers


class PseudoDemoGenerator:
    """
    Generator for pseudo-demonstrations with physics-aware object interaction.
    
    Creates procedurally generated trajectories with semantic consistency:
    - Each "task" consists of multiple demonstrations with similar motions
    - Objects are randomly placed but consistent across demos
    - Trajectories use motion primitives (pick-place, push, grasp) - 50%
    - Random waypoints for general skill acquisition - 50%
    - Constant step spacing: 1cm translation, 3 degrees rotation (Appendix D)
    - Objects follow gripper during grasping (attachment physics)
    
    This enables training the ICIL model without real task data.
    """
    
    def __init__(
        self,
        num_objects: int = 3,
        seed: Optional[int] = None,
        config: Optional[PseudoTaskConfig] = None,
    ):
        """
        Initialize the pseudo-demonstration generator.
        
        Args:
            num_objects: Number of objects in each scene
            seed: Random seed for reproducibility
            config: Full configuration (overrides other args)
        
        Note: trajectory_length is no longer a parameter - it is determined
        dynamically by the resampling process (Appendix D: 1cm and 3 degrees).
        """
        if config is not None:
            self.config = config
        else:
            self.config = PseudoTaskConfig(
                num_objects=num_objects,
            )
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.rng = np.random.RandomState(seed)
    
    def _select_motion_type(self) -> MotionType:
        """Select a motion type based on configured probabilities."""
        probs = self.config.motion_probs
        types = list(probs.keys())
        weights = [probs[t] for t in types]
        
        # Normalize
        total = sum(weights)
        weights = [w / total for w in weights]
        
        choice = self.rng.choice(len(types), p=weights)
        return MotionType(types[choice])
    
    def generate_task(
        self,
        num_demos: int = 2,
        motion_type: Optional[MotionType] = None,
    ) -> Dict:
        """
        Generate a pseudo-task with multiple semantically consistent demonstrations.
        
        Args:
            num_demos: Number of demonstrations per task
            motion_type: Specific motion type (random if None)
        
        Returns:
            Dictionary containing:
                - 'demos': List of demonstrations
                - 'scene': Scene configuration
                - 'motion_type': Type of motion used
        """
        # Select motion type
        if motion_type is None:
            motion_type = self._select_motion_type()
        
        # Generate scene (objects)
        scene = self._generate_scene()
        
        # Generate base trajectory using motion primitives
        base_trajectory = self._generate_base_trajectory(scene, motion_type)
        
        # Generate demonstrations as variations of base trajectory
        demos = []
        for i in range(num_demos):
            demo = self._generate_demo_variation(
                base_trajectory, 
                scene,
                variation_scale=0.1,
            )
            demos.append(demo)
        
        return {
            'demos': demos,
            'scene': scene,
            'motion_type': motion_type.value,
        }
    
    def _generate_scene(self) -> Dict:
        """
        Generate a random scene with objects.
        
        Returns:
            Scene dictionary with object positions, sizes, and point clouds
        """
        objects = []
        
        for i in range(self.config.num_objects):
            # Random position within workspace
            pos = np.array([
                self.config.workspace_center[0] + (self.rng.random() - 0.5) * self.config.workspace_size[0],
                self.config.workspace_center[1] + (self.rng.random() - 0.5) * self.config.workspace_size[1],
                self.config.workspace_center[2] + (self.rng.random() - 0.5) * self.config.workspace_size[2] * 0.3,  # Objects on lower part
            ])
            
            # Ensure objects are on/above table
            pos[2] = max(pos[2], self.config.workspace_center[2] - self.config.workspace_size[2] * 0.4)
            
            # Random size
            size = self.rng.uniform(*self.config.object_size_range)
            
            # Random shape (cube, sphere, cylinder)
            shape = self.rng.choice(['cube', 'sphere', 'cylinder'])
            
            # Generate point cloud for object
            pcd = self._generate_object_pcd(pos, size, shape)
            
            # Create ObjectState for physics tracking
            obj_state = ObjectState(pos, size, shape, pcd)
            
            objects.append({
                'position': pos,
                'size': size,
                'shape': shape,
                'pcd': pcd,
                'state': obj_state,
            })
        
        return {'objects': objects}
    
    def _generate_object_pcd(
        self,
        position: np.ndarray,
        size: float,
        shape: str,
    ) -> np.ndarray:
        """
        Generate point cloud for an object.
        
        Args:
            position: Object center position [3]
            size: Object size
            shape: Object shape type
        
        Returns:
            Point cloud [N, 3]
        """
        n_points = self.config.num_points_per_object
        
        if shape == 'cube':
            # Random points on cube surface
            faces = self.rng.randint(0, 6, n_points)
            points = np.zeros((n_points, 3))
            
            for i, face in enumerate(faces):
                axis = face // 2
                sign = (face % 2) * 2 - 1
                
                # Random point on face
                other_axes = [a for a in range(3) if a != axis]
                points[i, other_axes[0]] = (self.rng.random() - 0.5) * size
                points[i, other_axes[1]] = (self.rng.random() - 0.5) * size
                points[i, axis] = sign * size / 2
        
        elif shape == 'sphere':
            # Random points on sphere surface
            phi = self.rng.uniform(0, 2 * np.pi, n_points)
            theta = self.rng.uniform(0, np.pi, n_points)
            
            points = np.zeros((n_points, 3))
            points[:, 0] = size/2 * np.sin(theta) * np.cos(phi)
            points[:, 1] = size/2 * np.sin(theta) * np.sin(phi)
            points[:, 2] = size/2 * np.cos(theta)
        
        elif shape == 'cylinder':
            # Random points on cylinder surface
            angles = self.rng.uniform(0, 2 * np.pi, n_points)
            heights = (self.rng.random(n_points) - 0.5) * size
            
            points = np.zeros((n_points, 3))
            points[:, 0] = size/4 * np.cos(angles)
            points[:, 1] = size/4 * np.sin(angles)
            points[:, 2] = heights
        
        else:
            # Default to random blob
            points = (self.rng.random((n_points, 3)) - 0.5) * size
        
        # Translate to position
        points = points + position
        
        return points
    
    def _generate_base_trajectory(
        self,
        scene: Dict,
        motion_type: MotionType,
    ) -> Dict:
        """
        Generate a base trajectory using the appropriate motion primitive.
        
        After generating the raw trajectory, it is resampled to ensure constant
        spacing (Appendix D: 1cm translation and 3 degrees rotation).
        
        Args:
            scene: Scene configuration
            motion_type: Type of motion to generate
        
        Returns:
            Base trajectory dictionary with resampled trajectory
        """
        # Select target object (the one we interact with)
        target_obj = scene['objects'][0]
        target_pos = target_obj['position']
        target_size = target_obj['size']
        
        # Start position (above workspace)
        start_pos = np.array([
            self.config.workspace_center[0] + (self.rng.random() - 0.5) * 0.2,
            self.config.workspace_center[1] + (self.rng.random() - 0.5) * 0.2,
            self.config.workspace_center[2] + 0.2,
        ])
        
        # Grasp position (above target object)
        grasp_pos = target_pos.copy()
        grasp_pos[2] += target_size / 2 + 0.02
        
        if motion_type == MotionType.REACH_AND_GRASP:
            positions, rotations, grippers = MotionPrimitive.reach_and_grasp(
                start_pos=start_pos,
                target_pos=grasp_pos,
                rng=self.rng,
            )
        
        elif motion_type == MotionType.PICK_AND_PLACE:
            # Generate place position
            place_pos = np.array([
                self.config.workspace_center[0] + (self.rng.random() - 0.5) * 0.3,
                self.config.workspace_center[1] + (self.rng.random() - 0.5) * 0.3,
                target_pos[2],  # Same height as pick
            ])
            
            positions, rotations, grippers = MotionPrimitive.pick_and_place(
                start_pos=start_pos,
                pick_pos=grasp_pos,
                place_pos=place_pos,
                rng=self.rng,
            )
        
        elif motion_type == MotionType.PUSH:
            # Push direction
            push_dir = self.rng.randn(2)
            push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6) * 0.15
            
            push_start = target_pos.copy()
            push_start[2] = target_pos[2] - target_size * 0.3  # Push from side
            
            push_end = push_start.copy()
            push_end[:2] += push_dir
            
            positions, rotations, grippers = MotionPrimitive.push(
                start_pos=start_pos,
                push_start=push_start,
                push_end=push_end,
                rng=self.rng,
            )
        
        elif motion_type == MotionType.RANDOM_WAYPOINTS:
            # Random waypoints (Appendix D: "randomly select 2 to 6 waypoints")
            workspace_center = np.array(self.config.workspace_center)
            workspace_size = np.array(self.config.workspace_size)
            
            positions, rotations, grippers = MotionPrimitive.random_waypoints(
                workspace_center=workspace_center,
                workspace_size=workspace_size,
                rng=self.rng,
                min_waypoints=self.config.min_waypoints,
                max_waypoints=self.config.max_waypoints,
            )
        
        else:
            # Legacy fallback: random_spline
            end_pos = grasp_pos + (self.rng.random(3) - 0.5) * 0.1
            positions, rotations, grippers = MotionPrimitive.random_spline(
                start_pos=start_pos,
                end_pos=end_pos,
                rng=self.rng,
            )
        
        # === Resample trajectory to constant spacing (Appendix D) ===
        # "We ensure that the spacing between subsequent steps is constant 
        # and uniform (1cm and 3 degrees)"
        positions, rotations, grippers = resample_trajectory(
            positions=positions,
            rotations=rotations,
            grippers=grippers,
            trans_step=self.config.trans_step,
            rot_step=self.config.rot_step,
        )
        
        return {
            'positions': positions,
            'rotations': rotations,
            'grippers': grippers,
            'target_object': target_obj,
            'motion_type': motion_type,
        }
    
    def _generate_demo_variation(
        self,
        base_trajectory: Dict,
        scene: Dict,
        variation_scale: float = 0.1,
    ) -> Dict:
        """
        Generate a demonstration as a variation of the base trajectory.
        
        Implements:
        1. Object attachment: when gripper is closed and near an object,
           the object's point cloud follows the gripper's rigid body motion.
        2. Gripper state flipping: 10% chance to flip each gripper state
           (Appendix D: "for 10% of the data points, we deliberately change 
           the gripper's open/close state")
        
        Args:
            base_trajectory: Base trajectory to vary
            scene: Scene configuration
            variation_scale: Scale of random variations
        
        Returns:
            Demonstration dictionary with pcds, T_w_es, grips
        """
        length = len(base_trajectory['positions'])
        
        # Reset all object states
        for obj in scene['objects']:
            obj['state'].reset()
        
        # Add position noise
        positions = base_trajectory['positions'].copy()
        positions += self.rng.randn(*positions.shape) * self.config.noise_scale * variation_scale
        
        # Add rotation noise
        rotations = []
        for rot in base_trajectory['rotations']:
            noise_rot = R.from_rotvec(self.rng.randn(3) * 0.05 * variation_scale)
            rotations.append(noise_rot * rot)
        
        # === Gripper State Flipping (Appendix D) ===
        # "For 10% of the data points, we deliberately change the gripper's 
        # open/close state"
        grippers = list(base_trajectory['grippers'])
        for i in range(length):
            if self.rng.random() < self.config.gripper_flip_prob:
                grippers[i] = 1 - grippers[i]  # Flip: 0->1, 1->0
        
        # Build transforms and update object states
        T_w_es = []
        pcds = []
        
        for t in range(length):
            # Build gripper transform
            T = np.eye(4)
            T[:3, 3] = positions[t]
            T[:3, :3] = rotations[t].as_matrix()
            T_w_es.append(T)
            
            gripper_closed = grippers[t] == 0
            gripper_pos = positions[t]
            gripper_rot = rotations[t].as_matrix()
            
            # === Update Object States (Physics-aware) ===
            all_points = []
            for obj in scene['objects']:
                obj_state = obj['state']
                
                # Update object state based on gripper interaction
                obj_state.update(
                    gripper_position=gripper_pos,
                    gripper_rotation=gripper_rot,
                    gripper_closed=gripper_closed,
                    attachment_threshold=self.config.attachment_threshold,
                )
                
                # Get updated point cloud
                obj_pcd = obj_state.get_pcd()
                
                # Add small observation noise
                obj_pcd = obj_pcd + self.rng.randn(*obj_pcd.shape) * 0.001
                all_points.append(obj_pcd)
            
            # Combine all object point clouds
            pcd = np.vstack(all_points)
            pcds.append(pcd)
        
        return {
            'pcds': pcds,
            'T_w_es': T_w_es,
            'grips': grippers,
        }
    
    def save_task(self, task_data: Dict, path: str):
        """
        Save a generated task to disk.
        
        Note: ObjectState objects are not serializable, so we extract the
        necessary data for saving.
        
        Args:
            task_data: Task dictionary from generate_task()
            path: Path to save file
        """
        # Create a serializable version
        save_data = {
            'demos': task_data['demos'],
            'motion_type': task_data.get('motion_type'),
            'scene': {
                'objects': [
                    {
                        'position': obj['position'],
                        'size': obj['size'],
                        'shape': obj['shape'],
                        'pcd': obj['pcd'],
                    }
                    for obj in task_data['scene']['objects']
                ]
            }
        }
        torch.save(save_data, path)
    
    def load_task(self, path: str) -> Dict:
        """
        Load a task from disk.
        
        Args:
            path: Path to task file
        
        Returns:
            Task dictionary
        """
        return torch.load(path)
    
    def generate_dataset(
        self,
        output_dir: str,
        num_tasks: int,
        num_demos_per_task: int = 2,
        show_progress: bool = True,
    ):
        """
        Generate a full dataset of pseudo-demonstrations.
        
        Args:
            output_dir: Directory to save dataset
            num_tasks: Number of tasks to generate
            num_demos_per_task: Demonstrations per task
            show_progress: Whether to show progress bar
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Track motion type distribution
        motion_counts = {mt.value: 0 for mt in MotionType}
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(num_tasks), desc="Generating tasks")
            except ImportError:
                iterator = range(num_tasks)
                print("Install tqdm for progress bar: pip install tqdm")
        else:
            iterator = range(num_tasks)
        
        for i in iterator:
            task_data = self.generate_task(num_demos=num_demos_per_task)
            path = os.path.join(output_dir, f"task_{i:06d}.pt")
            self.save_task(task_data, path)
            
            # Track motion types
            motion_counts[task_data['motion_type']] += 1
        
        # Save metadata
        metadata = {
            'num_tasks': num_tasks,
            'num_demos_per_task': num_demos_per_task,
            'config': {
                k: v for k, v in self.config.__dict__.items()
                if not callable(v)
            },
            'motion_distribution': motion_counts,
        }
        torch.save(metadata, os.path.join(output_dir, 'metadata.pt'))
        
        print(f"Generated {num_tasks} tasks")
        print(f"Motion distribution: {motion_counts}")
