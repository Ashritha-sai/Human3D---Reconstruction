import torch


def pick_device(prefer: str = "cuda") -> str:
    prefer = (prefer or "cuda").lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
