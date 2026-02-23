"""
Utilities module for Human3D.

Provides common utility functions for configuration, device management,
and file I/O operations.

Functions:
    load_config: Load YAML configuration file
    pick_device: Select CUDA or CPU device
    read_bgr: Read and optionally resize BGR image
    ensure_dir: Create directory if it doesn't exist
    write_json: Write dictionary to JSON file
    save_png: Save image as PNG
    save_npy: Save NumPy array to .npy file
"""

from .config import load_config
from .device import pick_device
from .io import (
    read_bgr,
    ensure_dir,
    write_json,
    save_png,
    save_npy,
)

__all__ = [
    "load_config",
    "pick_device",
    "read_bgr",
    "ensure_dir",
    "write_json",
    "save_png",
    "save_npy",
]
