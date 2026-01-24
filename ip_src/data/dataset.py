"""
Dataset for Instant Policy Training.

Provides PyTorch Dataset implementations for loading pseudo-demonstrations
and preparing them for the graph-based flow matching model.

Key changes from standard diffusion policy datasets:
- Returns target_positions [horizon, 6, 3] for ghost gripper nodes
- NOT target_actions [horizon, 9] flat vectors
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from glob import glob

from ip_src.utils.sampling import sample_to_cond_demo, subsample_pcd, transform_pcd


class InstantPolicyDataset(Dataset):
    """
    Dataset for training Instant Policy.
    
    Loads pseudo-demonstrations and prepares them for the graph-based model.
    
    Each sample contains:
    - demo_pcds: Demonstration point clouds [num_demos, num_wp, num_points, 3]
    - demo_poses: Demonstration poses [num_demos, num_wp, 4, 4]
    - demo_grips: Demonstration gripper states [num_demos, num_wp]
    - live_pcd: Current point cloud [num_points, 3]
    - live_pose: Current pose [4, 4]
    - live_grip: Current gripper state
    - target_positions: Target ghost gripper positions [horizon, 6, 3]
    - target_grips: Target gripper states [horizon]
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
        num_waypoints: int = 10,
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
            num_waypoints: Number of waypoints per demonstration
            num_demos: Number of demonstrations per task
            prediction_horizon: Number of future keypoints to predict
            num_gripper_nodes: Number of gripper nodes (default 6)
            transform_to_ee_frame: Whether to transform point clouds to EE frame
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.num_waypoints = num_waypoints
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
        Get a training sample.
        
        Returns:
            Dictionary containing:
                - demo_pcds: [num_demos, num_wp, num_points, 3]
                - demo_poses: [num_demos, num_wp, 4, 4]
                - demo_grips: [num_demos, num_wp]
                - live_pcd: [num_points, 3]
                - live_pose: [4, 4]
                - live_grip: scalar
                - target_positions: [horizon, 6, 3] - Ghost gripper positions
                - target_grips: [horizon]
        """
        # Load task data
        task_data = torch.load(self.task_files[idx])
        demos = task_data['demos']
        
        # Ensure we have enough demos
        if len(demos) < self.num_demos:
            # Pad by repeating last demo
            while len(demos) < self.num_demos:
                demos.append(demos[-1])
        
        # Select demonstrations
        selected_demos = demos[:self.num_demos]
        
        # Process demonstrations
        processed_demos = []
        for demo in selected_demos:
            processed = sample_to_cond_demo(
                demo,
                num_waypoints=self.num_waypoints,
                num_points=self.num_points,
            )
            processed_demos.append(processed)
        
        # Stack demo data
        demo_pcds = []
        demo_poses = []
        demo_grips = []
        
        for demo in processed_demos:
            demo_pcds.append(np.stack(demo['obs']))
            demo_poses.append(np.stack(demo['T_w_es']))
            demo_grips.append(np.array(demo['grips']))
        
        demo_pcds = np.stack(demo_pcds)  # [num_demos, num_wp, num_points, 3]
        demo_poses = np.stack(demo_poses)  # [num_demos, num_wp, 4, 4]
        demo_grips = np.stack(demo_grips)  # [num_demos, num_wp]
        
        # Select a random timestep as "live" observation
        source_demo = demos[0]  # Use first demo as source for live/target
        traj_len = len(source_demo['pcds'])
        
        # Select a point where we have enough future steps for prediction
        max_start = max(1, traj_len - self.prediction_horizon - 1)
        live_idx = np.random.randint(0, max_start)
        
        # Get live observation
        live_pcd = source_demo['pcds'][live_idx]
        live_pose = source_demo['T_w_es'][live_idx]
        live_grip = source_demo['grips'][live_idx]
        
        # Process live point cloud
        live_pcd = subsample_pcd(live_pcd, num_points=self.num_points)
        if self.transform_to_ee_frame:
            T_e_w = np.linalg.inv(live_pose)
            live_pcd = transform_pcd(live_pcd, T_e_w)
        
        # === Compute target POSITIONS (not actions) ===
        # These are the future gripper node positions
        target_positions = []
        target_grips = []
        
        for i in range(self.prediction_horizon):
            future_idx = min(live_idx + i + 1, traj_len - 1)
            future_pose = source_demo['T_w_es'][future_idx]
            
            # Convert pose to 6 gripper node positions
            gripper_positions = self._pose_to_gripper_positions(future_pose)
            target_positions.append(gripper_positions)
            target_grips.append(source_demo['grips'][future_idx])
        
        target_positions = np.stack(target_positions)  # [horizon, 6, 3]
        target_grips = np.array(target_grips)  # [horizon]
        
        return {
            'demo_pcds': torch.tensor(demo_pcds, dtype=torch.float32),
            'demo_poses': torch.tensor(demo_poses, dtype=torch.float32),
            'demo_grips': torch.tensor(demo_grips, dtype=torch.float32),
            'live_pcd': torch.tensor(live_pcd, dtype=torch.float32),
            'live_pose': torch.tensor(live_pose, dtype=torch.float32),
            'live_grip': torch.tensor(live_grip, dtype=torch.float32),
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
