import numpy as np
from ultralytics import YOLO


class YOLOPose:
    def __init__(self, model_path: str, conf: float = 0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def predict(self, bgr: np.ndarray) -> dict:
        res = self.model.predict(source=bgr, conf=self.conf, verbose=False)
        r0 = res[0]

        out = {"people": []}

        if r0.boxes is None or len(r0.boxes) == 0:
            return out

        boxes = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = r0.boxes.conf.detach().cpu().numpy().astype(np.float32)

        kxy_all = None
        kcf_all = None
        if r0.keypoints is not None and r0.keypoints.xy is not None:
            kxy_all = r0.keypoints.xy.detach().cpu().numpy().astype(np.float32)
            if r0.keypoints.conf is not None:
                kcf_all = r0.keypoints.conf.detach().cpu().numpy().astype(np.float32)

        for i in range(len(boxes)):
            out["people"].append(
                {
                    "box_xyxy": boxes[i],
                    "conf": float(confs[i]),
                    "keypoints_xy": kxy_all[i] if kxy_all is not None else None,
                    "keypoints_conf": kcf_all[i] if kcf_all is not None else None,
                }
            )

        return out
