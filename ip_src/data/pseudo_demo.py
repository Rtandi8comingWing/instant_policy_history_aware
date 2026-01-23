"""
Pseudo-Demonstration Generator for Instant Policy.

Implements the pseudo-demonstration generation from Section 3.4 of the paper:
- Procedurally generates semantically consistent trajectory sets
- No real task semantics needed, just motion consistency
- Virtually infinite training data through simulation

Key features:
1. Object attachment: Objects follow gripper during grasping
2. Motion primitives: Pick-and-place, push, reach-and-grasp
3. Physics-aware point cloud generation

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
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from enum import Enum


class MotionType(Enum):
    """Types of motion primitives."""
    REACH_AND_GRASP = "reach_and_grasp"
    PICK_AND_PLACE = "pick_and_place"
    PUSH = "push"
    RANDOM_SPLINE = "random_spline"


@dataclass
class PseudoTaskConfig:
    """Configuration for pseudo-task generation."""
    num_objects: int = 3
    trajectory_length: int = 20
    workspace_size: Tuple[float, float, float] = (0.5, 0.5, 0.3)
    workspace_center: Tuple[float, float, float] = (0.5, 0.0, 0.2)
    object_size_range: Tuple[float, float] = (0.02, 0.08)
    num_points_per_object: int = 500
    gripper_action_prob: float = 0.2
    noise_scale: float = 0.01
    attachment_threshold: float = 0.05  # Distance threshold for object attachment
    # Motion primitive probabilities
    motion_probs: Dict[str, float] = None
    
    def __post_init__(self):
        if self.motion_probs is None:
            self.motion_probs = {
                MotionType.PICK_AND_PLACE.value: 0.4,
                MotionType.REACH_AND_GRASP.value: 0.3,
                MotionType.PUSH.value: 0.2,
                MotionType.RANDOM_SPLINE.value: 0.1,
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


class MotionPrimitive:
    """
    Motion primitives for generating diverse pseudo-demonstrations.
    
    Different motion patterns to create varied training data.
    """
    
    @staticmethod
    def reach_and_grasp(
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        length: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, List[R], List[int]]:
        """
        Generate reach-and-grasp motion.
        
        Motion: Approach target -> Grasp
        
        Args:
            start_pos: Starting position [3]
            target_pos: Target position [3]
            length: Trajectory length
            rng: Random number generator
        
        Returns:
            positions: [length, 3]
            rotations: List of Rotation objects
            grippers: List of gripper states (1=open, 0=closed)
        """
        positions = np.zeros((length, 3))
        
        # Approach phase (70% of trajectory)
        approach_len = int(length * 0.7)
        for i in range(approach_len):
            t = i / max(approach_len - 1, 1)
            positions[i] = start_pos + t * (target_pos - start_pos)
            # Add slight arc for more natural motion
            positions[i, 2] += 0.05 * np.sin(t * np.pi)
        
        # Grasp phase (30% of trajectory) - stay near target
        for i in range(approach_len, length):
            positions[i] = target_pos + rng.randn(3) * 0.002
        
        # Orientations - gripper pointing down
        base_rot = R.from_euler('xyz', [0, np.pi, 0])
        rotations = []
        for i in range(length):
            # Add small rotation noise
            noise_rot = R.from_rotvec(rng.randn(3) * 0.02)
            rotations.append(noise_rot * base_rot)
        
        # Grippers: open during approach, closed during grasp
        grippers = [1] * approach_len + [0] * (length - approach_len)
        
        return positions, rotations, grippers
    
    @staticmethod
    def pick_and_place(
        start_pos: np.ndarray,
        pick_pos: np.ndarray,
        place_pos: np.ndarray,
        length: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, List[R], List[int]]:
        """
        Generate pick-and-place motion.
        
        Motion: Approach -> Grasp -> Lift -> Move -> Lower -> Release
        
        Args:
            start_pos: Starting position [3]
            pick_pos: Pick position [3]
            place_pos: Place position [3]
            length: Trajectory length
            rng: Random number generator
        
        Returns:
            positions: [length, 3]
            rotations: List of Rotation objects
            grippers: List of gripper states
        """
        positions = np.zeros((length, 3))
        
        # Phase lengths
        approach_len = int(length * 0.2)
        grasp_len = int(length * 0.1)
        lift_len = int(length * 0.2)
        move_len = int(length * 0.3)
        lower_len = int(length * 0.1)
        release_len = length - approach_len - grasp_len - lift_len - move_len - lower_len
        
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
        while idx < length:
            positions[idx] = place_pos
            idx += 1
        
        # Orientations - gripper pointing down with slight variations
        base_rot = R.from_euler('xyz', [0, np.pi, 0])
        rotations = []
        for i in range(length):
            noise_rot = R.from_rotvec(rng.randn(3) * 0.02)
            rotations.append(noise_rot * base_rot)
        
        # Grippers
        grippers = (
            [1] * approach_len +          # Open during approach
            [0] * (grasp_len + lift_len + move_len + lower_len) +  # Closed while holding
            [1] * release_len             # Open to release
        )
        # Ensure correct length
        while len(grippers) < length:
            grippers.append(1)
        grippers = grippers[:length]
        
        return positions, rotations, grippers
    
    @staticmethod
    def push(
        start_pos: np.ndarray,
        push_start: np.ndarray,
        push_end: np.ndarray,
        length: int,
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, List[R], List[int]]:
        """
        Generate pushing motion.
        
        Motion: Approach -> Lower -> Push -> Lift
        
        Args:
            start_pos: Starting position [3]
            push_start: Where to start pushing [3]
            push_end: Where to end pushing [3]
            length: Trajectory length
            rng: Random number generator
        
        Returns:
            positions: [length, 3]
            rotations: List of Rotation objects
            grippers: List of gripper states
        """
        positions = np.zeros((length, 3))
        
        # Phase lengths
        approach_len = int(length * 0.25)
        lower_len = int(length * 0.1)
        push_len = int(length * 0.5)
        lift_len = length - approach_len - lower_len - push_len
        
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
        while idx < length:
            positions[idx] = lift_pos
            idx += 1
        
        # Orientations (tilted forward for pushing)
        push_dir = push_end - push_start
        push_dir[2] = 0  # Project to horizontal plane
        push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)
        
        rotations = []
        base_rot = R.from_euler('xyz', [0.3, np.pi, 0])  # Tilted forward
        for i in range(length):
            noise_rot = R.from_rotvec(rng.randn(3) * 0.02)
            rotations.append(noise_rot * base_rot)
        
        # Grippers (closed for pushing)
        grippers = [0] * length
        
        return positions, rotations, grippers
    
    @staticmethod
    def random_spline(
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        length: int,
        rng: np.random.RandomState,
        num_keypoints: int = 5,
    ) -> Tuple[np.ndarray, List[R], List[int]]:
        """
        Generate random spline trajectory (fallback).
        
        Args:
            start_pos: Starting position [3]
            end_pos: Ending position [3]
            length: Trajectory length
            rng: Random number generator
            num_keypoints: Number of spline keypoints
        
        Returns:
            positions: [length, 3]
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
        t_full = np.linspace(0, 1, length)
        
        positions = np.zeros((length, 3))
        for dim in range(3):
            cs = CubicSpline(t_keypoints, keypoints[:, dim])
            positions[:, dim] = cs(t_full)
        
        # Orientations
        base_rot = R.from_euler('xyz', [0, np.pi, 0])
        rotations = [base_rot] * length
        
        # Grippers: open at start, close at 70%
        close_point = int(length * 0.7)
        grippers = [1] * close_point + [0] * (length - close_point)
        
        return positions, rotations, grippers


class PseudoDemoGenerator:
    """
    Generator for pseudo-demonstrations with physics-aware object interaction.
    
    Creates procedurally generated trajectories with semantic consistency:
    - Each "task" consists of multiple demonstrations with similar motions
    - Objects are randomly placed but consistent across demos
    - Trajectories use motion primitives (pick-place, push, grasp)
    - Objects follow gripper during grasping (attachment physics)
    
    This enables training the ICIL model without real task data.
    """
    
    def __init__(
        self,
        num_objects: int = 3,
        trajectory_length: int = 20,
        seed: Optional[int] = None,
        config: Optional[PseudoTaskConfig] = None,
    ):
        """
        Initialize the pseudo-demonstration generator.
        
        Args:
            num_objects: Number of objects in each scene
            trajectory_length: Length of each trajectory
            seed: Random seed for reproducibility
            config: Full configuration (overrides other args)
        """
        if config is not None:
            self.config = config
        else:
            self.config = PseudoTaskConfig(
                num_objects=num_objects,
                trajectory_length=trajectory_length,
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
        
        Args:
            scene: Scene configuration
            motion_type: Type of motion to generate
        
        Returns:
            Base trajectory dictionary
        """
        length = self.config.trajectory_length
        
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
                length=length,
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
                length=length,
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
                length=length,
                rng=self.rng,
            )
        
        else:  # RANDOM_SPLINE
            end_pos = grasp_pos + (self.rng.random(3) - 0.5) * 0.1
            positions, rotations, grippers = MotionPrimitive.random_spline(
                start_pos=start_pos,
                end_pos=end_pos,
                length=length,
                rng=self.rng,
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
        
        Implements object attachment: when gripper is closed and near an object,
        the object's point cloud follows the gripper's rigid body motion.
        
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
        
        # Keep gripper states mostly the same
        grippers = list(base_trajectory['grippers'])
        
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
