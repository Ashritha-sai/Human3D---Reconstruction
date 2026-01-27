from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import numpy as np

from .utils.device import pick_device
from .utils.io import read_bgr, ensure_dir, save_png, save_npy, write_json
from .models.depth_midas import MiDaSDepth
from .models.pose_yolo import YOLOPose
from .reconstruct.pointcloud import depth_to_pointcloud, save_ply
from .viz.overlays import depth_to_vis, draw_pose, overlay_mask


@dataclass
class RunArtifacts:
    out_dir: Path


class Human3DPipeline:
    """
    Auto-mode behavior:
      - 0 people detected: skip pose/seg safely; depth/pointcloud still possible
      - 1 person detected: single-person mode
      - 2+ people detected: group mode (union of all subjects)

    Notes:
      - Expects YOLOPose.predict() to return:
          {"people": [{"box_xyxy": np.array([x1,y1,x2,y2]), "conf": float,
                       "keypoints_xy": (K,2) or None, "keypoints_conf": (K,) or None}, ...]}
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = pick_device(cfg.get("device", {}).get("prefer", "cuda"))

        self.depth = None
        if cfg.get("depth", {}).get("enabled", True):
            dcfg = cfg["depth"]
            self.depth = MiDaSDepth(dcfg.get("model_type", "DPT_Large"), self.device)

        self.pose = None
        if cfg.get("pose", {}).get("enabled", True):
            pcfg = cfg["pose"]
            self.pose = YOLOPose(
                pcfg.get("model", "yolov8n-pose.pt"), conf=float(pcfg.get("conf", 0.25))
            )

        self.seg = None
        if cfg.get("segmentation", {}).get("enabled", False):
            scfg = cfg["segmentation"]
            if scfg.get("method", "sam") == "sam":
                from .models.seg_sam import SAMSegmenter

                samcfg = scfg.get("sam", {})
                self.seg = SAMSegmenter(
                    checkpoint_path=samcfg["checkpoint_path"],
                    model_type=samcfg.get("model_type", "vit_b"),
                    device=self.device,
                )

    def _make_outdir(self) -> Path:
        base = self.cfg.get("run", {}).get("output_dir", "outputs")
        prefix = self.cfg.get("run", {}).get("name_prefix", "run")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return ensure_dir(Path(base) / f"{prefix}_{ts}")

    def _select_person_index(self, people: list, bgr_shape) -> int:
        """Selection policy used only when in single-person mode but multiple are present."""
        policy = self.cfg.get("subject", {}).get("selection", "highest_conf")
        idx_cfg = int(self.cfg.get("subject", {}).get("index", 0))
        H, W = bgr_shape[:2]

        if policy == "index":
            return int(np.clip(idx_cfg, 0, len(people) - 1))

        if policy == "highest_conf":
            return int(np.argmax([p["conf"] for p in people]))

        if policy == "largest_box":
            areas = []
            for p in people:
                x1, y1, x2, y2 = p["box_xyxy"]
                areas.append((x2 - x1) * (y2 - y1))
            return int(np.argmax(areas))

        if policy == "centered":
            cx_img, cy_img = W / 2.0, H / 2.0
            d2 = []
            for p in people:
                x1, y1, x2, y2 = p["box_xyxy"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                d2.append((cx - cx_img) ** 2 + (cy - cy_img) ** 2)
            return int(np.argmin(d2))

        return 0

    def _expand_clip_box(
        self, box_xyxy: np.ndarray, bgr_shape, expand_px: int
    ) -> np.ndarray:
        x1, y1, x2, y2 = box_xyxy.astype(np.float32)
        x1 -= expand_px
        y1 -= expand_px
        x2 += expand_px
        y2 += expand_px
        H, W = bgr_shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def run(self, input_path: str) -> str:
        out_dir = self._make_outdir()
        resize_max = self.cfg.get("input", {}).get("rgb_resize_max", 1280)
        bgr = read_bgr(input_path, resize_max=resize_max)

        summary = {
            "input": str(input_path),
            "device": self.device,
            "modules": {
                "depth": self.depth is not None,
                "pose": self.pose is not None,
                "segmentation": self.seg is not None,
            },
            "mode": None,
        }

        # -------------------------
        # Depth
        # -------------------------
        depth = None
        if self.depth is not None:
            depth = self.depth.predict(bgr)
            save_npy(out_dir / "depth.npy", depth)
            save_png(out_dir / "depth.png", depth_to_vis(depth))

        # -------------------------
        # Pose (detect all people)
        # -------------------------
        people = []
        if self.pose is not None:
            pose_out = self.pose.predict(bgr) or {}
            people = pose_out.get("people", []) or []

        summary["pose"] = {
            "num_people": int(len(people)),
        }

        # Decide auto-mode
        if len(people) == 0:
            summary["mode"] = "none"
        elif len(people) == 1:
            summary["mode"] = "single"
        else:
            # group photo -> all subjects
            summary["mode"] = "all"

        # -------------------------
        # Pose overlay output
        # -------------------------
        if len(people) > 0:
            if summary["mode"] == "single":
                sel = 0
                # if >1 but forced somehow, select based on policy
                if len(people) > 1:
                    sel = self._select_person_index(people, bgr.shape)

                p = people[sel]
                kxy = p.get("keypoints_xy", None)
                kcf = p.get("keypoints_conf", None)
                pose_img = draw_pose(bgr, kxy, kcf)

                save_png(out_dir / "pose_overlay.png", pose_img)
                summary["pose"]["selected_index"] = int(sel)
                summary["pose"]["selected_box_xyxy"] = p["box_xyxy"].tolist()

            else:
                # all mode: draw keypoints for all subjects
                pose_img = bgr.copy()
                for i, p in enumerate(people):
                    kxy = p.get("keypoints_xy", None)
                    kcf = p.get("keypoints_conf", None)
                    if kxy is not None:
                        pose_img = draw_pose(pose_img, kxy, kcf)
                save_png(out_dir / "pose_overlay.png", pose_img)
                summary["pose"]["selected_index"] = None
                summary["pose"]["selected_box_xyxy"] = None

        # -------------------------
        # Segmentation
        # -------------------------
        mask01 = None
        if self.seg is not None:
            seg_cfg = self.cfg.get("segmentation", {})
            prompt = seg_cfg.get("prompt", "box_from_pose")  # <-- correct location
            sam_cfg = seg_cfg.get("sam", {})
            expand = int(sam_cfg.get("expand_box_px", 30))

            if prompt != "box_from_pose":
                # fallback to full image segmentation if explicitly requested
                mask01 = self.seg.segment_full_image(bgr)

            else:
                # prompt == box_from_pose
                if len(people) == 0:
                    print("[WARN] No people detected; skipping SAM segmentation.")
                    mask01 = None
                elif summary["mode"] == "single":
                    # segment only the selected subject
                    sel = summary["pose"].get("selected_index", 0)
                    p = people[int(sel)]
                    box = self._expand_clip_box(
                        np.array(p["box_xyxy"]), bgr.shape, expand
                    )
                    mask01 = self.seg.segment_from_box(bgr, box)
                else:
                    # group mode: union masks from all boxes
                    masks = []
                    for p in people:
                        box = self._expand_clip_box(
                            np.array(p["box_xyxy"]), bgr.shape, expand
                        )
                        mi = self.seg.segment_from_box(bgr, box)
                        if mi is not None:
                            masks.append(mi.astype(bool))
                    if masks:
                        mask01 = np.logical_or.reduce(masks).astype(np.uint8)
                    else:
                        mask01 = None

            if mask01 is not None:
                # save mask/overlay
                save_png(out_dir / "seg_mask.png", (mask01 * 255).astype(np.uint8))
                save_png(out_dir / "seg_overlay.png", overlay_mask(bgr, mask01))
                summary["segmentation"] = {"mask_pixels": int(mask01.sum())}
            else:
                summary["segmentation"] = {"mask_pixels": 0}

        # -------------------------
        # 3D Pointcloud (optionally masked)
        # -------------------------
        if (
            self.cfg.get("reconstruction", {}).get("enabled", True)
            and depth is not None
        ):
            pcfg = self.cfg.get("reconstruction", {}).get("pointcloud", {})
            fx = float(pcfg.get("fx", 1000.0))
            fy = float(pcfg.get("fy", 1000.0))

            if mask01 is not None:
                # Mask depth to keep only subjects. This makes the point cloud cleaner.
                depth_masked = depth.copy()
                depth_masked[mask01 == 0] = 0.0
                pcd = depth_to_pointcloud(depth_masked, bgr, fx=fx, fy=fy)
            else:
                pcd = depth_to_pointcloud(depth, bgr, fx=fx, fy=fy)

            save_ply(str(out_dir / "pointcloud.ply"), pcd)

        write_json(out_dir / "summary.json", summary)
        return str(out_dir)
