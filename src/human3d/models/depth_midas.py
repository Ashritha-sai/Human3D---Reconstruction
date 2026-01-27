import numpy as np
import torch
import cv2


class MiDaSDepth:
    def __init__(self, model_type: str, device: str):
        self.device = device
        self.model_type = model_type
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(rgb).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth = pred.detach().cpu().numpy().astype(np.float32)
        return depth
