"""
Unit tests for Gaussian Splatting initialization.

Tests the depth unprojection and Gaussian parameter initialization
to ensure correct shapes, types, and value ranges.
"""

import sys
from pathlib import Path
import importlib.util

import numpy as np
import torch
import pytest

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# =============================================================================
# Module Import Helpers (to avoid dependency chain issues)
# =============================================================================


def _import_gaussian_utils():
    """Import gaussian_utils directly to avoid package dependencies."""
    spec = importlib.util.spec_from_file_location(
        "gaussian_utils",
        Path(__file__).parent.parent
        / "src"
        / "human3d"
        / "reconstruct"
        / "gaussian_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_pointcloud():
    """Import pointcloud directly to avoid package dependencies."""
    spec = importlib.util.spec_from_file_location(
        "pointcloud",
        Path(__file__).parent.parent
        / "src"
        / "human3d"
        / "reconstruct"
        / "pointcloud.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_gaussian_trainer():
    """Import gaussian_trainer directly to avoid package dependencies."""
    # First import gaussian_utils since trainer depends on it
    gaussian_utils = _import_gaussian_utils()

    # Set up a proper module structure that dataclass can work with
    import types

    # Create parent modules if they don't exist
    if "human3d" not in sys.modules:
        human3d = types.ModuleType("human3d")
        sys.modules["human3d"] = human3d

    if "human3d.reconstruct" not in sys.modules:
        reconstruct = types.ModuleType("human3d.reconstruct")
        sys.modules["human3d.reconstruct"] = reconstruct
        sys.modules["human3d"].reconstruct = reconstruct

    # Add gaussian_utils to the module hierarchy
    sys.modules["human3d.reconstruct.gaussian_utils"] = gaussian_utils
    sys.modules["human3d.reconstruct"].gaussian_utils = gaussian_utils

    # Create a proper module for gaussian_trainer
    trainer_path = (
        Path(__file__).parent.parent
        / "src"
        / "human3d"
        / "reconstruct"
        / "gaussian_trainer.py"
    )
    spec = importlib.util.spec_from_file_location(
        "human3d.reconstruct.gaussian_trainer", trainer_path
    )
    module = importlib.util.module_from_spec(spec)

    # Register it before loading so dataclass can find it
    sys.modules["human3d.reconstruct.gaussian_trainer"] = module

    spec.loader.exec_module(module)
    return module


# =============================================================================
# Tests for depth_to_xyz
# =============================================================================


class TestDepthToXYZ:
    """Tests for the depth_to_xyz function."""

    def test_basic_unprojection(self):
        """Test basic depth unprojection with a flat plane."""
        gu = _import_gaussian_utils()

        # Create a 64x64 depth map with constant depth = 2.0
        h, w = 64, 64
        depth = np.full((h, w), 2.0, dtype=np.float32)
        mask = np.ones((h, w), dtype=np.uint8)

        # Camera parameters
        fx, fy = 500.0, 500.0
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        xyz = gu.depth_to_xyz(depth, mask, fx, fy, cx, cy)

        # Should have h * w points
        assert xyz.shape == (h * w, 3), f"Expected ({h * w}, 3), got {xyz.shape}"
        assert xyz.dtype == np.float32

        # All Z values should be 2.0
        np.testing.assert_allclose(xyz[:, 2], 2.0, rtol=1e-5)

        # Center pixel should be at (0, 0, 2)
        center_idx = (h // 2) * w + (w // 2)
        np.testing.assert_allclose(xyz[center_idx], [0, 0, 2.0], atol=0.1)

    def test_masked_unprojection(self):
        """Test that mask correctly filters points."""
        gu = _import_gaussian_utils()

        h, w = 64, 64
        depth = np.full((h, w), 2.0, dtype=np.float32)

        # Create circular mask in center
        mask = np.zeros((h, w), dtype=np.uint8)
        cy, cx = h // 2, w // 2
        radius = 10
        y, x = np.ogrid[:h, :w]
        mask_circle = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        mask[mask_circle] = 1

        expected_points = mask.sum()

        xyz = gu.depth_to_xyz(depth, mask, 500.0, 500.0, (w - 1) / 2, (h - 1) / 2)

        assert xyz.shape[0] == expected_points, (
            f"Expected {expected_points} points, got {xyz.shape[0]}"
        )

    def test_torch_tensor_input(self):
        """Test that function works with PyTorch tensors."""
        gu = _import_gaussian_utils()

        h, w = 32, 32
        depth = torch.full((h, w), 1.5, dtype=torch.float32)
        mask = torch.ones((h, w), dtype=torch.float32)

        xyz = gu.depth_to_xyz(depth, mask, 500.0, 500.0, (w - 1) / 2, (h - 1) / 2)

        assert isinstance(xyz, torch.Tensor)
        assert xyz.shape == (h * w, 3)
        assert xyz.dtype == torch.float32

    def test_zero_depth_filtered(self):
        """Test that zero depth values are filtered out."""
        gu = _import_gaussian_utils()

        h, w = 32, 32
        depth = np.full((h, w), 2.0, dtype=np.float32)
        depth[0:10, 0:10] = 0.0  # Zero out a region
        mask = np.ones((h, w), dtype=np.uint8)

        xyz = gu.depth_to_xyz(depth, mask, 500.0, 500.0, (w - 1) / 2, (h - 1) / 2)

        # Should have fewer points than h * w
        expected = h * w - 10 * 10
        assert xyz.shape[0] == expected, (
            f"Expected {expected} points, got {xyz.shape[0]}"
        )


# =============================================================================
# Tests for normalize_depth_to_metric
# =============================================================================


class TestNormalizeDepth:
    """Tests for depth normalization."""

    def test_normalization_range(self):
        """Test that normalized depth is in expected range."""
        gu = _import_gaussian_utils()

        # Random depth values
        depth = np.random.rand(64, 64).astype(np.float32) * 100

        normalized = gu.normalize_depth_to_metric(depth, min_depth=0.5, max_depth=2.5)

        assert normalized.min() >= 0.5 - 1e-6
        assert normalized.max() <= 2.5 + 1e-6

    def test_normalization_torch(self):
        """Test normalization with PyTorch tensors."""
        gu = _import_gaussian_utils()

        depth = torch.rand(64, 64) * 100

        normalized = gu.normalize_depth_to_metric(depth, min_depth=0.5, max_depth=2.5)

        assert isinstance(normalized, torch.Tensor)
        assert normalized.min() >= 0.5 - 1e-6
        assert normalized.max() <= 2.5 + 1e-6


# =============================================================================
# Tests for estimate_point_scales
# =============================================================================


class TestEstimatePointScales:
    """Tests for scale estimation."""

    def test_basic_scale_estimation(self):
        """Test scale estimation on random points."""
        gu = _import_gaussian_utils()

        # Create 100 random points
        xyz = np.random.rand(100, 3).astype(np.float32)

        scales = gu.estimate_point_scales(xyz, k_neighbors=8)

        assert scales.shape == (100, 3)
        assert scales.dtype == np.float32
        assert (scales > 0).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="pytorch3d KNN requires CUDA or specific setup",
    )
    def test_scale_estimation_torch(self):
        """Test scale estimation with PyTorch tensors."""
        gu = _import_gaussian_utils()

        xyz = torch.rand(100, 3, dtype=torch.float32)

        scales = gu.estimate_point_scales(xyz, k_neighbors=8)

        assert isinstance(scales, torch.Tensor)
        assert scales.shape == (100, 3)
        assert (scales > 0).all()


# =============================================================================
# Tests for RGB to SH conversion
# =============================================================================


class TestRGBToSH:
    """Tests for RGB to spherical harmonics conversion."""

    def test_sh_degree_0(self):
        """Test SH conversion with degree 0 (DC only)."""
        gu = _import_gaussian_utils()

        rgb = np.array(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
            ],
            dtype=np.float32,
        )

        sh = gu.rgb_to_spherical_harmonics(rgb, degree=0)

        assert sh.shape == (3, 1, 3)
        assert sh.dtype == np.float32

    def test_sh_degree_3(self):
        """Test SH conversion with degree 3 (full)."""
        gu = _import_gaussian_utils()

        rgb = np.random.rand(10, 3).astype(np.float32)

        sh = gu.rgb_to_spherical_harmonics(rgb, degree=3)

        # (degree + 1)^2 = 16 coefficients
        assert sh.shape == (10, 16, 3)

    def test_sh_roundtrip(self):
        """Test that RGB -> SH -> RGB preserves colors."""
        gu = _import_gaussian_utils()

        rgb = np.random.rand(10, 3).astype(np.float32)

        sh = gu.rgb_to_spherical_harmonics(rgb, degree=0)
        rgb_recovered = gu.spherical_harmonics_to_rgb(sh)

        # Use slightly relaxed tolerance for float32 precision
        np.testing.assert_allclose(rgb, rgb_recovered, rtol=1e-4, atol=1e-6)


# =============================================================================
# Tests for GaussianTrainer
# =============================================================================


class TestGaussianTrainerInit:
    """Tests for GaussianTrainer initialization."""

    def create_synthetic_data(self, h=512, w=512):
        """Create synthetic test data."""
        # RGB image - random colors
        rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        # Depth map - plane at z=2.0 with some variation
        depth = np.full((h, w), 2.0, dtype=np.float32)
        depth += np.random.randn(h, w).astype(np.float32) * 0.1

        # Circular mask in center
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).astype(np.uint8)

        return rgb, depth, mask

    def test_trainer_init(self):
        """Test basic trainer initialization."""
        gt = _import_gaussian_trainer()

        rgb, depth, mask = self.create_synthetic_data(256, 256)

        camera = gt.CameraParams(
            fx=1000.0,
            fy=1000.0,
            cx=127.5,
            cy=127.5,
            width=256,
            height=256,
        )
        config = gt.GaussianConfig(sh_degree=3)

        trainer = gt.GaussianTrainer(
            rgb=rgb,
            depth=depth,
            mask=mask,
            camera_params=camera,
            config=config,
            device="cpu",
        )

        assert trainer.rgb.shape == (256, 256, 3)
        assert trainer.depth.shape == (256, 256)
        assert trainer.mask.shape == (256, 256)
        assert trainer.num_gaussians == 0  # Before initialization

    def test_initialize_gaussians(self):
        """Test Gaussian initialization from depth map."""
        gt = _import_gaussian_trainer()

        h, w = 512, 512
        rgb, depth, mask = self.create_synthetic_data(h, w)

        camera = gt.CameraParams(
            fx=1000.0,
            fy=1000.0,
            cx=(w - 1) / 2.0,
            cy=(h - 1) / 2.0,
            width=w,
            height=h,
        )
        config = gt.GaussianConfig(sh_degree=3, opacity_init=0.9)

        trainer = gt.GaussianTrainer(
            rgb=rgb,
            depth=depth,
            mask=mask,
            camera_params=camera,
            config=config,
            device="cpu",
        )

        num_gaussians = trainer.initialize_gaussians()

        # Check number of Gaussians matches masked pixels
        expected_points = mask.sum()
        assert num_gaussians == expected_points, (
            f"Expected {expected_points}, got {num_gaussians}"
        )

        # Check parameter shapes
        assert trainer.means.shape == (num_gaussians, 3)
        assert trainer.scales.shape == (num_gaussians, 3)
        assert trainer.rotations.shape == (num_gaussians, 4)
        assert trainer.sh_coeffs.shape == (num_gaussians, 16, 3)  # degree 3 = 16 coeffs
        assert trainer.opacities.shape == (num_gaussians,)

        # Check that means are within expected Z range (after normalization: 0.5 to 2.5)
        z_values = trainer.means[:, 2].detach().numpy()
        assert z_values.min() >= 0.4, f"Z min {z_values.min()} too low"
        assert z_values.max() <= 2.6, f"Z max {z_values.max()} too high"

        # Check rotations are identity quaternions
        assert torch.allclose(trainer.rotations[:, 0], torch.ones(num_gaussians))
        assert torch.allclose(trainer.rotations[:, 1:], torch.zeros(num_gaussians, 3))

        # Check optimizer is set up
        assert trainer.optimizer is not None

    def test_get_parameters(self):
        """Test get_parameters returns correct structure."""
        gt = _import_gaussian_trainer()

        rgb, depth, mask = self.create_synthetic_data(128, 128)

        camera = gt.CameraParams(
            fx=500.0, fy=500.0, cx=63.5, cy=63.5, width=128, height=128
        )
        config = gt.GaussianConfig(sh_degree=0)

        trainer = gt.GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        params = trainer.get_parameters()

        assert "means" in params
        assert "scales" in params
        assert "rotations" in params
        assert "sh_coeffs" in params
        assert "opacities" in params

        # All should be tensors
        for name, tensor in params.items():
            assert isinstance(tensor, torch.Tensor), f"{name} is not a Tensor"
            assert tensor.requires_grad, f"{name} does not require grad"


# =============================================================================
# Tests for consistency with original implementation
# =============================================================================


class TestConsistencyWithOriginal:
    """Tests to verify consistency with original pointcloud.py implementation."""

    @pytest.mark.skipif(
        not importlib.util.find_spec("open3d"), reason="open3d not installed"
    )
    def test_xyz_matches_original(self):
        """Verify depth_to_xyz produces same results as original."""
        gu = _import_gaussian_utils()
        pc = _import_pointcloud()

        # Create test data
        h, w = 64, 64
        depth_raw = np.random.rand(h, w).astype(np.float32) * 10
        bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.ones((h, w), dtype=np.uint8)

        fx, fy = 1000.0, 1000.0
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        # Original implementation
        pcd = pc.depth_to_pointcloud(depth_raw, bgr, fx, fy)
        xyz_original = np.asarray(pcd.points)

        # New implementation
        depth_normalized = gu.normalize_depth_to_metric(
            depth_raw, min_depth=0.5, max_depth=2.5
        )
        xyz_new = gu.depth_to_xyz(depth_normalized, mask, fx, fy, cx, cy)

        # Should have same number of points (both use full image)
        assert xyz_original.shape[0] == xyz_new.shape[0], (
            f"Point count mismatch: original {xyz_original.shape[0]} vs new {xyz_new.shape[0]}"
        )

        # Sort both arrays for comparison (order may differ)
        xyz_original_sorted = xyz_original[np.lexsort(xyz_original.T)]
        xyz_new_sorted = xyz_new[np.lexsort(xyz_new.T)]

        # Values should be very close
        np.testing.assert_allclose(
            xyz_original_sorted,
            xyz_new_sorted,
            rtol=1e-4,
            atol=1e-6,
            err_msg="XYZ coordinates don't match original implementation",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
