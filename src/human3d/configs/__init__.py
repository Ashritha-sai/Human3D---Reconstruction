"""
Configuration module for Human3D.

Contains default configuration files for various pipeline components.

Files:
    gaussian_config.yaml: Hyperparameters for Gaussian Splatting training
"""

from pathlib import Path

# Path to the configs directory
CONFIGS_DIR = Path(__file__).parent

# Paths to specific config files
GAUSSIAN_CONFIG_PATH = CONFIGS_DIR / "gaussian_config.yaml"


def get_default_gaussian_config_path() -> Path:
    """
    Get the path to the default Gaussian splatting config file.

    Returns:
        Path: Absolute path to gaussian_config.yaml
    """
    return GAUSSIAN_CONFIG_PATH
