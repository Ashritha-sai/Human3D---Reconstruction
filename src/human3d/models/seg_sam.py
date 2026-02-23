import os
import numpy as np
import cv2


class SAMSegmenter:
    def __init__(self, checkpoint_path: str, model_type: str, device: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}\n"
                "Download a SAM .pth and place it there, or set segmentation.enabled=false."
            )
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def segment_from_box(self, bgr: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)
        masks, scores, _ = self.predictor.predict(
            box=box_xyxy.astype(np.float32), multimask_output=True
        )
        best = int(np.argmax(scores))
        return masks[best].astype(np.uint8)  # (H,W) 0/1

    def segment_full_image(self, bgr: np.ndarray) -> np.ndarray:
        # Fallback: one big box over entire image
        h, w = bgr.shape[:2]
        box = np.array([0, 0, w - 1, h - 1], dtype=np.float32)
        return self.segment_from_box(bgr, box)
