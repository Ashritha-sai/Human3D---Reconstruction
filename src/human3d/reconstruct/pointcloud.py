import numpy as np
import open3d as o3d
import cv2

from .gaussian_utils import normalize_depth_to_metric, depth_to_xyz


def depth_to_pointcloud(
    depth: np.ndarray, bgr: np.ndarray, fx: float, fy: float
) -> o3d.geometry.PointCloud:
    h, w = depth.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    # Normalize depth to pseudo-metric range
    z = normalize_depth_to_metric(depth, min_depth=0.5, max_depth=2.5)

    # Unproject to 3D
    mask = np.ones((h, w), dtype=np.uint8)
    pts = depth_to_xyz(z, mask, fx, fy, cx, cy)

    # Get RGB colors for valid points
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32) / 255.0

    # Filter same way as depth_to_xyz (valid = mask & z > 0 & finite)
    valid = np.isfinite(z).reshape(-1) & (z.reshape(-1) > 0)
    rgb = rgb[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def save_ply(path: str, pcd: o3d.geometry.PointCloud) -> None:
    o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=False)
