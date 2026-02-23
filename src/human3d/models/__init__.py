"""
Models module for Human3D.

Provides neural network model wrappers for depth estimation,
pose detection, and segmentation.

Classes:
    MiDaSDepth: MiDaS monocular depth estimation wrapper
    YOLOPose: YOLOv8 pose detection wrapper
    SAMSegmenter: Segment Anything Model wrapper
"""

from .depth_midas import MiDaSDepth
from .pose_yolo import YOLOPose
from .seg_sam import SAMSegmenter

__all__ = [
    "MiDaSDepth",
    "YOLOPose",
    "SAMSegmenter",
]
