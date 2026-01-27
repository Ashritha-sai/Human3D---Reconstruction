import numpy as np
import open3d as o3d
import cv2


def depth_to_pointcloud(
    depth: np.ndarray, bgr: np.ndarray, fx: float, fy: float
) -> o3d.geometry.PointCloud:
    h, w = depth.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    # Normalize depth to a pseudo-metric range (coarse but consistent)
    d = depth.copy()
    d = d - np.nanmin(d)
    denom = np.nanmax(d) + 1e-6
    d = d / denom
    z = 0.5 + 2.0 * d  # ~0.5m to 2.5m-ish fake scale

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    Z = z

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32) / 255.0

    # drop invalid / extreme
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    rgb = rgb[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def save_ply(path: str, pcd: o3d.geometry.PointCloud) -> None:
    o3d.io.write_point_cloud(path, pcd, write_ascii=False, compressed=False)
