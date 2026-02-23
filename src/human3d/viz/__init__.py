"""
Visualization module for Human3D.

Provides functions for visualizing depth maps, poses, and segmentation masks.

Functions:
    depth_to_vis: Convert depth map to colorized visualization
    draw_pose: Draw pose keypoints on image
    overlay_mask: Overlay segmentation mask on image
"""

from .overlays import (
    depth_to_vis,
    draw_pose,
    overlay_mask,
)

__all__ = [
    "depth_to_vis",
    "draw_pose",
    "overlay_mask",
]
