"""
Dataset for Instant Policy Training.

Provides PyTorch Dataset implementations for loading pseudo-demonstrations
and preparing them for the graph-based flow matching model.

Implementation follows Appendix E (Implementation Details):
- Point clouds are transformed to End-Effector (EE) frame for spatial generalization
- Dense trajectories (1cm spacing) are smart-downsampled to L=10 for context
- Keyframes (start, end, gripper changes) are prioritized during downsampling
- Target poses remain dense (1cm spacing) for accurate prediction

Key data flow:
- Context (demos): Sparse L=10 waypoints, point clouds in local EE frame
- Live: Current observation, point cloud in local EE frame  
- Target: Dense horizon frames (1cm spacing), poses in world frame
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
from glob import glob

from ip_src.utils.sampling import subsample_pcd


class InstantPolicyDataset(Dataset):
    """
    Dataset for training Instant Policy (Appendix E compliant).
    
    Loads dense pseudo-demonstrations and prepares them for the graph-based model
    with proper coordinate transformations and smart downsampling.
    
    Key features per Appendix E:
    1. Point clouds transformed to EE frame (P_local = T_WE^{-1} @ P_world)
    2. Context demos downsampled to L=10 with keyframe preservation
    3. Target poses remain dense (1cm spacing) for accurate prediction
    
    Each sample contains:
    - demo_pcds: [num_demos, context_len, num_points, 3] - Local EE frame
    - demo_poses: [num_demos, context_len, 4, 4] - World frame (for graph edges)
    - demo_grips: [num_demos, context_len] - Gripper states
    - live_pcd: [num_points, 3] - Local EE frame
    - live_pose: [4, 4] - World frame
    - live_grip: scalar
    - target_poses: [horizon, 4, 4] - World frame, DENSE 1cm spacing
    - target_positions: [horizon, 6, 3] - Ghost gripper positions
    - target_grips: [horizon]
    """
    
    # Gripper node offsets (same as in GraphBuilder)
    GRIPPER_OFFSETS = np.array([
        [0.0, 0.04, 0.0],   # Left finger tip
        [0.0, -0.04, 0.0],  # Right finger tip
        [0.05, 0.0, 0.0],   # Forward (x-axis)
        [-0.03, 0.0, 0.0],  # Backward
        [0.0, 0.0, 0.03],   # Up (z-axis)
        [0.0, 0.0, -0.03],  # Down
    ])
    
    def __init__(
        self,
        data_dir: str,
        num_points: int = 2048,
        context_len: int = 10,
        num_demos: int = 2,
        prediction_horizon: int = 8,
        num_gripper_nodes: int = 6,
        transform_to_ee_frame: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing task files
            num_points: Number of points per point cloud
            context_len: Number of waypoints per demo context (default 10, per Appendix E)
            num_demos: Number of demonstrations per task
            prediction_horizon: Number of future dense steps to predict
            num_gripper_nodes: Number of gripper nodes (default 6)
            transform_to_ee_frame: Whether to transform point clouds to EE frame
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.context_len = context_len  # L=10 per Appendix E
        self.num_demos = num_demos
        self.prediction_horizon = prediction_horizon
        self.num_gripper_nodes = num_gripper_nodes
        self.transform_to_ee_frame = transform_to_ee_frame
        
        # Find all task files
        self.task_files = sorted(glob(os.path.join(data_dir, "task_*.pt")))
        
        if len(self.task_files) == 0:
            raise ValueError(f"No task files found in {data_dir}")
        
        # Load metadata if available
        metadata_path = os.path.join(data_dir, "metadata.pt")
        if os.path.exists(metadata_path):
            self.metadata = torch.load(metadata_path)
        else:
            self.metadata = None
    
    def __len__(self) -> int:
        return len(self.task_files)
    
    def _to_local(
        self,
        pcd_world: Union[np.ndarray, torch.Tensor],
        pose: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform point cloud from world frame to local EE frame (Appendix E).
        
        P_local = T_WE^{-1} @ P_world
        
        This is critical for spatial generalization - the model learns
        object positions relative to the gripper, not absolute world positions.
        
        Args:
            pcd_world: Point cloud in world coordinates [N, 3]
            pose: End-effector pose T_WE [4, 4]
        
        Returns:
            pcd_local: Point cloud in EE local coordinates [N, 3]
        """
        use_torch = isinstance(pcd_world, torch.Tensor)
        
        if use_torch:
            # Torch implementation
            T_EW = torch.linalg.inv(pose)  # [4, 4]
            
            # Convert to homogeneous coordinates
            ones = torch.ones(pcd_world.shape[0], 1, 
                            dtype=pcd_world.dtype, device=pcd_world.device)
            pcd_homo = torch.cat([pcd_world, ones], dim=-1)  # [N, 4]
            
            # Transform: P_local = T_EW @ P_world
            pcd_local_homo = (T_EW @ pcd_homo.T).T  # [N, 4]
            pcd_local = pcd_local_homo[:, :3]  # [N, 3]
        else:
            # NumPy implementation
            T_EW = np.linalg.inv(pose)  # [4, 4]
            
            # Convert to homogeneous coordinates
            ones = np.ones((pcd_world.shape[0], 1))
            pcd_homo = np.concatenate([pcd_world, ones], axis=-1)  # [N, 4]
            
            # Transform: P_local = T_EW @ P_world
            pcd_local_homo = (T_EW @ pcd_homo.T).T  # [N, 4]
            pcd_local = pcd_local_homo[:, :3]  # [N, 3]
        
        return pcd_local
    
    def _smart_downsample(
        self,
        num_points: int,
        grippers: Union[np.ndarray, List[int]],
        target_len: int = 10,
    ) -> np.ndarray:
        """
        Smart downsampling of dense trajectory to fixed length (Appendix E).
        
        Prioritizes keyframes in the following order:
        1. Start and end points (indices 0 and N-1)
        2. Gripper state change points (critical for manipulation)
        3. Linear interpolation to fill remaining slots
        
        This preserves the semantically important frames while reducing
        the context length for efficient transformer processing.
        
        Args:
            num_points: Total number of points in dense trajectory
            grippers: Gripper states for each timestep [num_points]
            target_len: Target number of samples (default 10)
        
        Returns:
            indices: Sorted array of indices [target_len]
        """
        if num_points <= target_len:
            # If trajectory is already short enough, use all points
            # and pad with last index if needed
            indices = list(range(num_points))
            while len(indices) < target_len:
                indices.append(num_points - 1)
            return np.array(indices)
        
        grippers = np.array(grippers)
        keyframe_indices = set()
        
        # Priority 1: Start and end points (always include)
        keyframe_indices.add(0)
        keyframe_indices.add(num_points - 1)
        
        # Priority 2: Gripper state change points
        # Find where gripper state changes (open->close or close->open)
        grip_diff = np.diff(grippers)
        change_points = np.where(grip_diff != 0)[0] + 1  # +1 because diff shifts index
        
        for cp in change_points:
            if len(keyframe_indices) < target_len:
                keyframe_indices.add(cp)
                # Also add the frame just before the change
                if cp > 0:
                    keyframe_indices.add(cp - 1)
        
        # Priority 3: Fill remaining with linear interpolation
        keyframe_indices = sorted(keyframe_indices)
        
        if len(keyframe_indices) >= target_len:
            # Too many keyframes, sample uniformly from them
            step = len(keyframe_indices) / target_len
            indices = [keyframe_indices[int(i * step)] for i in range(target_len)]
        else:
            # Need to add more points via interpolation
            indices = list(keyframe_indices)
            remaining = target_len - len(indices)
            
            # Create candidate points (excluding already selected)
            all_indices = set(range(num_points))
            available = sorted(all_indices - set(indices))
            
            if remaining > 0 and len(available) > 0:
                # Uniformly sample from available indices
                step = len(available) / remaining
                for i in range(remaining):
                    idx = available[min(int(i * step), len(available) - 1)]
                    indices.append(idx)
        
        # Sort indices to maintain temporal order
        indices = sorted(set(indices))
        
        # Ensure exactly target_len indices
        if len(indices) > target_len:
            # Uniformly subsample
            step = len(indices) / target_len
            indices = [indices[int(i * step)] for i in range(target_len)]
        elif len(indices) < target_len:
            # Pad with last index
            while len(indices) < target_len:
                indices.append(indices[-1])
        
        return np.array(indices)
    
    def _pose_to_gripper_positions(self, pose: np.ndarray) -> np.ndarray:
        """
        Convert pose matrix to 6 gripper node positions.
        
        Args:
            pose: SE(3) pose matrix [4, 4]
        
        Returns:
            Gripper node positions [6, 3]
        """
        position = pose[:3, 3]
        rotation = pose[:3, :3]
        
        # Transform offsets to world frame
        world_offsets = (rotation @ self.GRIPPER_OFFSETS.T).T  # [6, 3]
        positions = position + world_offsets  # [6, 3]
        
        return positions
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training sample with proper coordinate transformations.
        
        Data flow (per Appendix E):
        1. Load dense trajectory (1cm spacing from pseudo_demo)
        2. Context: Smart downsample to L=10, transform pcds to EE frame
        3. Live: Random timestep t, transform pcd to EE frame
        4. Target: Dense frames t+1 to t+horizon (keep 1cm spacing!)
        
        Returns:
            Dictionary containing:
                - demo_pcds: [num_demos, context_len, num_points, 3] - LOCAL EE frame
                - demo_poses: [num_demos, context_len, 4, 4] - World frame
                - demo_grips: [num_demos, context_len]
                - live_pcd: [num_points, 3] - LOCAL EE frame
                - live_pose: [4, 4] - World frame
                - live_grip: scalar
                - target_poses: [horizon, 4, 4] - World frame, DENSE
                - target_positions: [horizon, 6, 3] - Ghost gripper positions
                - target_grips: [horizon]
        """
        # Load task data (contains dense 1cm-spaced trajectories)
        task_data = torch.load(self.task_files[idx])
        demos = task_data['demos']
        
        # Ensure we have enough demos
        if len(demos) < self.num_demos:
            while len(demos) < self.num_demos:
                demos.append(demos[-1])
        
        # Select demos for context
        selected_demos = demos[:self.num_demos]
        
        # ============================================================
        # CONTEXT: Process multiple demos, smart downsample to L=10
        # ============================================================
        all_demo_pcds = []
        all_demo_poses = []
        all_demo_grips = []
        
        for demo in selected_demos:
            dense_pcds = demo['pcds']
            dense_poses = demo['T_w_es']
            dense_grips = demo['grips']
            traj_len = len(dense_pcds)
            
            # Smart downsample this demo
            downsample_indices = self._smart_downsample(
                num_points=traj_len,
                grippers=dense_grips,
                target_len=self.context_len,
            )
            
            # Extract sparse context from dense trajectory
            demo_pcds = []
            demo_poses = []
            demo_grips = []
            
            for i in downsample_indices:
                pcd = np.array(dense_pcds[i])
                pose = np.array(dense_poses[i])
                grip = dense_grips[i]
                
                # Subsample point cloud to fixed size
                pcd = subsample_pcd(pcd, num_points=self.num_points)
                
                # Transform point cloud to LOCAL EE frame (Appendix E)
                if self.transform_to_ee_frame:
                    pcd = self._to_local(pcd, pose)
                
                demo_pcds.append(pcd)
                demo_poses.append(pose)  # Keep pose in world frame for graph edges
                demo_grips.append(grip)
            
            all_demo_pcds.append(np.stack(demo_pcds))
            all_demo_poses.append(np.stack(demo_poses))
            all_demo_grips.append(np.array(demo_grips))
        
        # Stack all demos: [num_demos, context_len, ...]
        demo_pcds = np.stack(all_demo_pcds)    # [num_demos, context_len, num_points, 3]
        demo_poses = np.stack(all_demo_poses)  # [num_demos, context_len, 4, 4]
        demo_grips = np.stack(all_demo_grips)  # [num_demos, context_len]
        
        # ============================================================
        # LIVE: Random timestep from first demo's dense trajectory
        # ============================================================
        source_demo = demos[0]
        dense_pcds = source_demo['pcds']
        dense_poses = source_demo['T_w_es']
        dense_grips = source_demo['grips']
        traj_len = len(dense_pcds)
        
        # Select a point where we have enough future steps for prediction
        max_start = max(1, traj_len - self.prediction_horizon - 1)
        live_idx = np.random.randint(0, max_start)
        
        live_pcd = np.array(dense_pcds[live_idx])
        live_pose = np.array(dense_poses[live_idx])
        live_grip = dense_grips[live_idx]
        
        # Subsample and transform to LOCAL EE frame
        live_pcd = subsample_pcd(live_pcd, num_points=self.num_points)
        if self.transform_to_ee_frame:
            live_pcd = self._to_local(live_pcd, live_pose)
        
        # ============================================================
        # TARGET: Dense future frames (KEEP 1cm spacing!)
        # Critical: Do NOT use downsampled indices for target!
        # ============================================================
        target_poses = []
        target_positions = []
        target_grips = []
        
        for i in range(self.prediction_horizon):
            # Use dense indices: live_idx + 1, live_idx + 2, ...
            future_idx = min(live_idx + i + 1, traj_len - 1)
            future_pose = np.array(dense_poses[future_idx])
            
            # Store target pose in WORLD frame (for SE(3) flow matching)
            target_poses.append(future_pose)
            
            # Convert pose to 6 gripper node positions
            gripper_positions = self._pose_to_gripper_positions(future_pose)
            target_positions.append(gripper_positions)
            target_grips.append(dense_grips[future_idx])
        
        target_poses = np.stack(target_poses)      # [horizon, 4, 4]
        target_positions = np.stack(target_positions)  # [horizon, 6, 3]
        target_grips = np.array(target_grips)      # [horizon]
        
        return {
            # Context (demos) - point clouds in LOCAL EE frame
            'demo_pcds': torch.tensor(demo_pcds, dtype=torch.float32),
            'demo_poses': torch.tensor(demo_poses, dtype=torch.float32),
            'demo_grips': torch.tensor(demo_grips, dtype=torch.float32),
            # Live observation - point cloud in LOCAL EE frame
            'live_pcd': torch.tensor(live_pcd, dtype=torch.float32),
            'live_pose': torch.tensor(live_pose, dtype=torch.float32),
            'live_grip': torch.tensor(live_grip, dtype=torch.float32),
            # Target (prediction) - DENSE 1cm spacing, WORLD frame
            'target_poses': torch.tensor(target_poses, dtype=torch.float32),
            'target_positions': torch.tensor(target_positions, dtype=torch.float32),
            'target_grips': torch.tensor(target_grips, dtype=torch.float32),
        }


class RLBenchDataset(Dataset):
    """
    Dataset for RLBench demonstrations.
    
    Loads real demonstrations from RLBench for evaluation or fine-tuning.
    """
    
    GRIPPER_OFFSETS = InstantPolicyDataset.GRIPPER_OFFSETS
    
    def __init__(
        self,
        data_dir: str,
        task_name: str,
        num_points: int = 2048,
        num_waypoints: int = 10,
        num_demos: int = 2,
        prediction_horizon: int = 8,
        camera_names: Tuple[str, ...] = ('front', 'left_shoulder', 'right_shoulder'),
    ):
        self.data_dir = data_dir
        self.task_name = task_name
        self.num_points = num_points
        self.num_waypoints = num_waypoints
        self.num_demos = num_demos
        self.prediction_horizon = prediction_horizon
        self.camera_names = camera_names
        
        # Load demonstration files
        task_dir = os.path.join(data_dir, task_name)
        self.demo_files = sorted(glob(os.path.join(task_dir, "episode_*.pt")))
        
        if len(self.demo_files) < num_demos:
            raise ValueError(
                f"Not enough demonstrations for task {task_name}. "
                f"Found {len(self.demo_files)}, need {num_demos}"
            )
    
    def __len__(self) -> int:
        return len(self.demo_files)
    
    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError(
            "RLBenchDataset requires RLBench installation. "
            "Use InstantPolicyDataset with pseudo-demonstrations instead."
        )


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching samples.
    """
    result = {}
    
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            result[key] = torch.tensor(np.stack(values))
        else:
            result[key] = values
    
    return result
