"""
Sampling and Data Processing Utilities for Instant Policy.

Implements the sample_to_cond_demo function and related utilities
for processing demonstrations into the format expected by the model.

Compatible with the original instant_policy.pyi interface.
"""

import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Tuple, Union


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


def downsample_pcd(pcd: np.ndarray, voxel_size: float = 0.01) -> np.ndarray:
    """
    Voxel downsample point cloud.
    
    Args:
        pcd: Point cloud [N, 3]
        voxel_size: Voxel size
    
    Returns:
        Downsampled point cloud [M, 3]
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size)
    return np.asarray(pcd_o3d.points)


def remove_statistical_outliers(
    pcd: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove statistical outliers from point cloud.
    
    Args:
        pcd: Point cloud [N, 3]
        nb_neighbors: Number of neighbors to consider
        std_ratio: Standard deviation ratio threshold
    
    Returns:
        filtered_pcd: Filtered point cloud [M, 3]
        inlier_indices: Indices of inlier points [M]
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    
    filtered_pcd, inlier_indices = pcd_o3d.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    
    return np.asarray(filtered_pcd.points), np.array(inlier_indices)


def compute_demo_actions(
    T_w_es: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Compute relative actions between consecutive waypoints.
    
    Args:
        T_w_es: List of end-effector transforms [T][4, 4]
    
    Returns:
        List of relative transforms [T-1][4, 4]
    """
    actions = []
    for i in range(len(T_w_es) - 1):
        T_curr = T_w_es[i]
        T_next = T_w_es[i + 1]
        # Relative transform: T_rel = T_curr^{-1} @ T_next
        T_rel = np.linalg.inv(T_curr) @ T_next
        actions.append(T_rel)
    return actions


def interpolate_trajectory(
    T_w_es: List[np.ndarray],
    num_points: int,
) -> List[np.ndarray]:
    """
    Interpolate trajectory to have exactly num_points waypoints.
    
    Uses linear interpolation for translation and SLERP for rotation.
    
    Args:
        T_w_es: List of transforms [T][4, 4]
        num_points: Target number of waypoints
    
    Returns:
        Interpolated transforms [num_points][4, 4]
    """
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp
    
    if len(T_w_es) == num_points:
        return T_w_es
    
    # Extract translations and rotations
    translations = np.array([T[:3, 3] for T in T_w_es])
    rotations = R.from_matrix([T[:3, :3] for T in T_w_es])
    
    # Original timestamps
    t_orig = np.linspace(0, 1, len(T_w_es))
    # Target timestamps
    t_target = np.linspace(0, 1, num_points)
    
    # Interpolate translations
    interp_trans = np.zeros((num_points, 3))
    for i in range(3):
        interp_trans[:, i] = np.interp(t_target, t_orig, translations[:, i])
    
    # Interpolate rotations using SLERP
    slerp = Slerp(t_orig, rotations)
    interp_rots = slerp(t_target)
    
    # Combine into transforms
    result = []
    for i in range(num_points):
        T = np.eye(4)
        T[:3, 3] = interp_trans[i]
        T[:3, :3] = interp_rots[i].as_matrix()
        result.append(T)
    
    return result


def segment_point_cloud(
    pcd: np.ndarray,
    colors: Optional[np.ndarray] = None,
    method: str = "dbscan",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment point cloud into clusters.
    
    Args:
        pcd: Point cloud [N, 3]
        colors: Optional point colors [N, 3]
        method: Segmentation method ("dbscan", "ransac_plane")
        **kwargs: Method-specific parameters
    
    Returns:
        labels: Cluster labels [N]
        centroids: Cluster centroids [K, 3]
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    
    if colors is not None:
        pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    
    if method == "dbscan":
        eps = kwargs.get("eps", 0.02)
        min_points = kwargs.get("min_points", 10)
        labels = np.array(pcd_o3d.cluster_dbscan(eps=eps, min_points=min_points))
    
    elif method == "ransac_plane":
        # Remove dominant plane (e.g., table)
        distance_threshold = kwargs.get("distance_threshold", 0.01)
        ransac_n = kwargs.get("ransac_n", 3)
        num_iterations = kwargs.get("num_iterations", 1000)
        
        _, inliers = pcd_o3d.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        
        labels = np.zeros(len(pcd), dtype=int)
        labels[inliers] = -1  # Mark plane points
        # Cluster remaining points
        remaining = np.ones(len(pcd), dtype=bool)
        remaining[inliers] = False
        
        if remaining.sum() > 0:
            pcd_remaining = pcd_o3d.select_by_index(np.where(remaining)[0])
            cluster_labels = np.array(
                pcd_remaining.cluster_dbscan(
                    eps=kwargs.get("eps", 0.02),
                    min_points=kwargs.get("min_points", 10),
                )
            )
            labels[remaining] = cluster_labels + 1
    
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    
    # Compute centroids
    unique_labels = np.unique(labels[labels >= 0])
    centroids = np.zeros((len(unique_labels), 3))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        centroids[i] = pcd[mask].mean(axis=0)
    
    return labels, centroids
