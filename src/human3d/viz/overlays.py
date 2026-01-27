import numpy as np
import cv2


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    d = depth.copy()
    d = d - np.nanmin(d)
    d = d / (np.nanmax(d) + 1e-6)
    vis = (255.0 * d).clip(0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    return vis


def draw_pose(
    bgr: np.ndarray,
    keypoints_xy: np.ndarray | None,
    keypoints_conf: np.ndarray | None = None,
) -> np.ndarray:
    out = bgr.copy()
    if keypoints_xy is None:
        return out
    for i, (x, y) in enumerate(keypoints_xy):
        if keypoints_conf is not None and keypoints_conf[i] < 0.2:
            continue
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    return out


def overlay_mask(bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    if mask01 is None:
        return out
    mask = mask01.astype(np.uint8) * 255
    color = np.zeros_like(out)
    color[:, :, 1] = 180  # green overlay
    alpha = 0.45
    m3 = cv2.merge([mask, mask, mask])
    out = np.where(m3 > 0, (out * (1 - alpha) + color * alpha).astype(np.uint8), out)
    return out
