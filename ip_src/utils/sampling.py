"""
Sampling and Data Processing Utilities for Instant Policy.

Implements the sample_to_cond_demo function and related utilities
for processing demonstrations into the format expected by the model.

Compatible with the original instant_policy.pyi interface.
"""

import numpy as np
import open3d as o3d
from typing import Dict, List


def sample_to_cond_demo(
    sample: Dict,
    num_waypoints: int,
    num_points: int = 2048,
) -> Dict:
    """
    Convert raw demonstration to conditioning format.
    
    This function is compatible with the original instant_policy interface.
    
    It performs:
    1. Uniform temporal sampling of waypoints
    2. Point cloud downsampling and outlier removal
    3. Transformation to end-effector frame
    
    Args:
        sample: Dictionary containing:
            - 'pcds': List of point clouds in world frame [T][N, 3]
            - 'T_w_es': List of end-effector transforms [T][4, 4]
            - 'grips': List of gripper states [T] (0=closed, 1=open)
        num_waypoints: Number of waypoints to sample
        num_points: Number of points per point cloud
    
    Returns:
        Dictionary containing:
            - 'obs': List of point clouds in EE frame [num_waypoints][num_points, 3]
            - 'T_w_es': List of transforms [num_waypoints][4, 4]
            - 'grips': List of gripper states [num_waypoints]
    """
    pcds = sample['pcds']
    T_w_es = sample['T_w_es']
    grips = sample['grips']
    
    total_steps = len(pcds)
    
    # Uniform temporal sampling
    if total_steps <= num_waypoints:
        # If we have fewer steps than waypoints, use all steps
        indices = list(range(total_steps))
        # Pad to num_waypoints by repeating last step
        while len(indices) < num_waypoints:
            indices.append(total_steps - 1)
    else:
        # Uniform sampling
        indices = np.linspace(0, total_steps - 1, num_waypoints, dtype=int).tolist()
    
    # Process each sampled waypoint
    obs_list = []
    T_w_es_list = []
    grips_list = []
    
    for idx in indices:
        pcd_w = pcds[idx]  # Point cloud in world frame
        T_w_e = T_w_es[idx]  # World to end-effector transform
        grip = grips[idx]
        
        # Subsample and clean point cloud
        pcd_subsampled = subsample_pcd(pcd_w, num_points=num_points)
        
        # Transform to end-effector frame
        T_e_w = np.linalg.inv(T_w_e)
        pcd_e = transform_pcd(pcd_subsampled, T_e_w)
        
        obs_list.append(pcd_e)
        T_w_es_list.append(T_w_e)
        grips_list.append(grip)
    
    return {
        'obs': obs_list,
        'T_w_es': T_w_es_list,
        'grips': grips_list,
    }


def subsample_pcd(
    pcd: np.ndarray,
    num_points: int = 2048,
    voxel_size: float = 0.01,
    remove_outliers: bool = True,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """
    Subsample point cloud to fixed number of points.
    
    Performs:
    1. Voxel downsampling
    2. Statistical outlier removal (optional)
    3. Random sampling to exact number of points
    
    Args:
        pcd: Input point cloud [N, 3]
        num_points: Target number of points
        voxel_size: Voxel size for downsampling
        remove_outliers: Whether to remove outliers
        nb_neighbors: Number of neighbors for outlier detection
        std_ratio: Standard deviation ratio for outlier detection
    
    Returns:
        Subsampled point cloud [num_points, 3]
    """
    # Convert to Open3D point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    
    # Voxel downsampling
    pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size)
    
    # Statistical outlier removal
    if remove_outliers and len(pcd_o3d.points) > nb_neighbors:
        pcd_o3d, _ = pcd_o3d.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
    
    # Convert back to numpy
    pcd_np = np.asarray(pcd_o3d.points)
    
    # Random sampling to exact number of points
    n_points = len(pcd_np)
    if n_points == 0:
        # Handle empty point cloud
        return np.zeros((num_points, 3))
    elif n_points < num_points:
        # Upsample by repeating points
        indices = np.random.choice(n_points, num_points, replace=True)
    else:
        # Downsample by random selection
        indices = np.random.choice(n_points, num_points, replace=False)
    
    return pcd_np[indices]


def transform_pcd(pcd: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform point cloud by SE(3) transformation.
    
    Args:
        pcd: Point cloud [N, 3]
        T: Transformation matrix [4, 4]
    
    Returns:
        Transformed point cloud [N, 3]
    """
    return np.matmul(T[:3, :3], pcd.T).T + T[:3, 3]
