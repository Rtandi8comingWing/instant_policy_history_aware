"""
SE(3) Transform Utilities for Instant Policy.

Implements transformation utilities including:
- 6D rotation representation (for continuous rotation learning)
- Pose/transform conversions
- Point cloud transformations

Based on "On the Continuity of Rotation Representations in Neural Networks" (CVPR 2019)
"""

import torch
import numpy as np
from typing import Union
from scipy.spatial.transform import Rotation as Rot


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix.
    
    Uses the first two columns and Gram-Schmidt orthogonalization.
    Reference: "On the Continuity of Rotation Representations in Neural Networks"
    
    Args:
        rot_6d: 6D rotation representation [..., 6]
    
    Returns:
        Rotation matrix [..., 3, 3]
    """
    # Extract first two columns
    a1 = rot_6d[..., :3]  # [..., 3]
    a2 = rot_6d[..., 3:6]  # [..., 3]
    
    # Gram-Schmidt orthogonalization
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    
    # Remove component parallel to b1 from a2
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    
    # Third column via cross product
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Stack into matrix
    matrix = torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3]
    
    return matrix


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D representation.
    
    Simply takes the first two columns of the rotation matrix.
    
    Args:
        matrix: Rotation matrix [..., 3, 3]
    
    Returns:
        6D rotation representation [..., 6]
    """
    # Take first two columns
    col1 = matrix[..., :, 0]  # [..., 3]
    col2 = matrix[..., :, 1]  # [..., 3]
    
    return torch.cat([col1, col2], dim=-1)  # [..., 6]


def pose_to_transform(pose: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert pose (position + quaternion) to 4x4 transformation matrix.
    
    Args:
        pose: [7] array/tensor with [x, y, z, qx, qy, qz, qw]
    
    Returns:
        4x4 transformation matrix
    """
    if isinstance(pose, torch.Tensor):
        return _pose_to_transform_torch(pose)
    else:
        return _pose_to_transform_numpy(pose)


def _pose_to_transform_numpy(pose: np.ndarray) -> np.ndarray:
    """NumPy implementation of pose to transform."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rot.from_quat(pose[3:]).as_matrix()
    return T


def _pose_to_transform_torch(pose: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of pose to transform."""
    T = torch.eye(4, device=pose.device, dtype=pose.dtype)
    T[:3, 3] = pose[:3]
    T[:3, :3] = quaternion_to_matrix(pose[3:])
    return T


def transform_to_pose(T: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert 4x4 transformation matrix to pose (position + quaternion).
    
    Args:
        T: 4x4 transformation matrix
    
    Returns:
        [7] array/tensor with [x, y, z, qx, qy, qz, qw]
    """
    if isinstance(T, torch.Tensor):
        return _transform_to_pose_torch(T)
    else:
        return _transform_to_pose_numpy(T)


def _transform_to_pose_numpy(T: np.ndarray) -> np.ndarray:
    """NumPy implementation of transform to pose."""
    pose = np.zeros(7)
    pose[:3] = T[:3, 3]
    pose[3:] = Rot.from_matrix(T[:3, :3]).as_quat()
    return pose


def _transform_to_pose_torch(T: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of transform to pose."""
    pose = torch.zeros(7, device=T.device, dtype=T.dtype)
    pose[:3] = T[:3, 3]
    pose[3:] = matrix_to_quaternion(T[:3, :3])
    return pose


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        quaternion: [qx, qy, qz, qw] format [4]
    
    Returns:
        Rotation matrix [3, 3]
    """
    qx, qy, qz, qw = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    
    # Normalize quaternion
    norm = torch.sqrt(qx*qx + qy*qy + qz*qz + qw*qw + 1e-8)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Build rotation matrix
    matrix = torch.zeros(3, 3, device=quaternion.device, dtype=quaternion.dtype)
    
    matrix[0, 0] = 1 - 2*(qy*qy + qz*qz)
    matrix[0, 1] = 2*(qx*qy - qz*qw)
    matrix[0, 2] = 2*(qx*qz + qy*qw)
    
    matrix[1, 0] = 2*(qx*qy + qz*qw)
    matrix[1, 1] = 1 - 2*(qx*qx + qz*qz)
    matrix[1, 2] = 2*(qy*qz - qx*qw)
    
    matrix[2, 0] = 2*(qx*qz - qy*qw)
    matrix[2, 1] = 2*(qy*qz + qx*qw)
    matrix[2, 2] = 1 - 2*(qx*qx + qy*qy)
    
    return matrix


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        matrix: Rotation matrix [3, 3]
    
    Returns:
        Quaternion [qx, qy, qz, qw] format [4]
    """
    # Shepperd's method for numerical stability
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (matrix[2, 1] - matrix[1, 2]) * s
        y = (matrix[0, 2] - matrix[2, 0]) * s
        z = (matrix[1, 0] - matrix[0, 1]) * s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    
    return torch.tensor([x, y, z, w], device=matrix.device, dtype=matrix.dtype)


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
