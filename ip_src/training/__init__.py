"""
Training components for Instant Policy.
"""

from ip_src.training.trainer import InstantPolicyTrainer, EMACallback
from ip_src.training.losses import FlowLoss, GripperLoss, CombinedLoss

__all__ = [
    "InstantPolicyTrainer",
    "EMACallback",
    "FlowLoss",
    "GripperLoss",
    "CombinedLoss",
]
