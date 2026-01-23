"""
Dataset for Instant Policy Training.

Provides PyTorch Dataset implementations for loading pseudo-demonstrations
and real demonstrations.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from glob import glob

from ip_src.utils.sampling import sample_to_cond_demo, subsample_pcd, transform_pcd
from ip_src.utils.transforms import (
    matrix_to_rotation_6d,
    compute_relative_transform,
    transform_to_action,
)


class InstantPolicyDataset(Dataset):
    """
    Dataset for training Instant Policy.
    
    Loads pseudo-demonstrations and prepares them for training.
    Each sample contains:
    - Multiple demonstration trajectories (context)
    - Target actions to predict
    """
    
    def __init__(
        self,
        data_dir: str,
        num_points: int = 2048,
        num_waypoints: int = 10,
        num_demos: int = 2,
        prediction_horizon: int = 8,
        transform_to_ee_frame: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing task files
            num_points: Number of points per point cloud
            num_waypoints: Number of waypoints per demonstration
            num_demos: Number of demonstrations per task
            prediction_horizon: Number of future actions to predict
            transform_to_ee_frame: Whether to transform point clouds to EE frame
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.num_waypoints = num_waypoints
        self.num_demos = num_demos
        self.prediction_horizon = prediction_horizon
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
                - target_actions: [horizon, 9]
                - target_grips: [horizon]
        """
        # Load task data
        task_data = torch.load(self.task_files[idx])
        demos = task_data['demos']
        
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
        # Use one of the demonstration trajectories for the live observation
        source_demo = demos[0]  # Use first demo as source for live
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
        
        # Compute target actions (relative transforms)
        target_actions = []
        target_grips = []
        
        for i in range(self.prediction_horizon):
            future_idx = min(live_idx + i + 1, traj_len - 1)
            
            if i == 0:
                curr_pose = live_pose
            else:
                curr_pose = source_demo['T_w_es'][live_idx + i]
            
            next_pose = source_demo['T_w_es'][future_idx]
            
            # Compute relative transform
            T_rel = np.linalg.inv(curr_pose) @ next_pose
            
            # Convert to action representation (translation + 6D rotation)
            translation = T_rel[:3, 3]
            rotation_6d = self._matrix_to_6d(T_rel[:3, :3])
            action = np.concatenate([translation, rotation_6d])
            
            target_actions.append(action)
            target_grips.append(source_demo['grips'][future_idx])
        
        target_actions = np.stack(target_actions)  # [horizon, 9]
        target_grips = np.array(target_grips)  # [horizon]
        
        return {
            'demo_pcds': torch.tensor(demo_pcds, dtype=torch.float32),
            'demo_poses': torch.tensor(demo_poses, dtype=torch.float32),
            'demo_grips': torch.tensor(demo_grips, dtype=torch.float32),
            'live_pcd': torch.tensor(live_pcd, dtype=torch.float32),
            'live_pose': torch.tensor(live_pose, dtype=torch.float32),
            'live_grip': torch.tensor(live_grip, dtype=torch.float32),
            'target_actions': torch.tensor(target_actions, dtype=torch.float32),
            'target_grips': torch.tensor(target_grips, dtype=torch.float32),
        }
    
    def _matrix_to_6d(self, matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to 6D representation."""
        return matrix[:, :2].flatten()


class RLBenchDataset(Dataset):
    """
    Dataset for RLBench demonstrations.
    
    Loads real demonstrations from RLBench for evaluation or fine-tuning.
    """
    
    def __init__(
        self,
        data_dir: str,
        task_name: str,
        num_points: int = 2048,
        num_waypoints: int = 10,
        num_demos: int = 2,
        camera_names: Tuple[str, ...] = ('front', 'left_shoulder', 'right_shoulder'),
    ):
        """
        Initialize RLBench dataset.
        
        Args:
            data_dir: Directory containing RLBench data
            task_name: Name of the task
            num_points: Number of points per point cloud
            num_waypoints: Number of waypoints per demonstration
            num_demos: Number of demonstrations to use as context
            camera_names: Camera names for point cloud extraction
        """
        self.data_dir = data_dir
        self.task_name = task_name
        self.num_points = num_points
        self.num_waypoints = num_waypoints
        self.num_demos = num_demos
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
        """Get a training sample from RLBench data."""
        # Implementation would follow similar pattern to InstantPolicyDataset
        # but loading from RLBench-specific format
        raise NotImplementedError(
            "RLBenchDataset requires RLBench installation. "
            "Use InstantPolicyDataset with pseudo-demonstrations instead."
        )


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching samples.
    
    Args:
        batch: List of sample dictionaries
    
    Returns:
        Batched dictionary
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
