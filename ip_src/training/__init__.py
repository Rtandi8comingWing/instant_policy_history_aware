"""
Training components for Instant Policy.
"""

from ip_src.training.trainer import InstantPolicyTrainer
from ip_src.training.losses import DiffusionLoss, GripperLoss, CombinedLoss

__all__ = [
    "InstantPolicyTrainer",
    "DiffusionLoss",
    "GripperLoss",
    "CombinedLoss",
]
