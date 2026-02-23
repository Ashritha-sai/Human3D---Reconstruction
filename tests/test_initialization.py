"""
Unit tests for Gaussian Splatting initialization.

Tests the depth unprojection and Gaussian parameter initialization
to ensure correct shapes, types, and value ranges.
"""

import importlib.util

import numpy as np
import torch
import pytest

from human3d.reconstruct import gaussian_utils as gu
from human3d.reconstruct.gaussian_trainer import GaussianTrainer, GaussianConfig, CameraParams


# =============================================================================
# Tests for depth_to_xyz
# =============================================================================


class TestDepthToXYZ:
    """Tests for the depth_to_xyz function."""

    def test_basic_unprojection(self):
        """Test basic depth unprojection with a flat plane."""
        h, w = 64, 64
        depth = np.full((h, w), 2.0, dtype=np.float32)
        mask = np.ones((h, w), dtype=np.uint8)

        fx, fy = 500.0, 500.0
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        xyz = gu.depth_to_xyz(depth, mask, fx, fy, cx, cy)

        assert xyz.shape == (h * w, 3), f"Expected ({h * w}, 3), got {xyz.shape}"
        assert xyz.dtype == np.float32

        np.testing.assert_allclose(xyz[:, 2], 2.0, rtol=1e-5)

        center_idx = (h // 2) * w + (w // 2)
        np.testing.assert_allclose(xyz[center_idx], [0, 0, 2.0], atol=0.1)

    def test_masked_unprojection(self):
        """Test that mask correctly filters points."""
        h, w = 64, 64
        depth = np.full((h, w), 2.0, dtype=np.float32)

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
        h, w = 32, 32
        depth = torch.full((h, w), 1.5, dtype=torch.float32)
        mask = torch.ones((h, w), dtype=torch.float32)

        xyz = gu.depth_to_xyz(depth, mask, 500.0, 500.0, (w - 1) / 2, (h - 1) / 2)

        assert isinstance(xyz, torch.Tensor)
        assert xyz.shape == (h * w, 3)
        assert xyz.dtype == torch.float32

    def test_zero_depth_filtered(self):
        """Test that zero depth values are filtered out."""
        h, w = 32, 32
        depth = np.full((h, w), 2.0, dtype=np.float32)
        depth[0:10, 0:10] = 0.0
        mask = np.ones((h, w), dtype=np.uint8)

        xyz = gu.depth_to_xyz(depth, mask, 500.0, 500.0, (w - 1) / 2, (h - 1) / 2)

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
        depth = np.random.rand(64, 64).astype(np.float32) * 100

        normalized = gu.normalize_depth_to_metric(depth, min_depth=0.5, max_depth=2.5)

        assert normalized.min() >= 0.5 - 1e-6
        assert normalized.max() <= 2.5 + 1e-6

    def test_normalization_torch(self):
        """Test normalization with PyTorch tensors."""
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
        xyz = np.random.rand(100, 3).astype(np.float32)

        scales = gu.estimate_point_scales(xyz, k_neighbors=8)

        assert scales.shape == (100, 3)
        assert scales.dtype == np.float32
        assert (scales > 0).all()

    def test_scale_estimation_torch(self):
        """Test scale estimation with PyTorch tensors."""
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
        rgb = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        sh = gu.rgb_to_spherical_harmonics(rgb, degree=0)

        assert sh.shape == (3, 1, 3)
        assert sh.dtype == np.float32

    def test_sh_degree_3(self):
        """Test SH conversion with degree 3 (full)."""
        rgb = np.random.rand(10, 3).astype(np.float32)

        sh = gu.rgb_to_spherical_harmonics(rgb, degree=3)

        assert sh.shape == (10, 16, 3)

    def test_sh_roundtrip(self):
        """Test that RGB -> SH -> RGB preserves colors."""
        rgb = np.random.rand(10, 3).astype(np.float32)

        sh = gu.rgb_to_spherical_harmonics(rgb, degree=0)
        rgb_recovered = gu.spherical_harmonics_to_rgb(sh)

        np.testing.assert_allclose(rgb, rgb_recovered, rtol=1e-4, atol=1e-6)


# =============================================================================
# Tests for GaussianTrainer
# =============================================================================


class TestGaussianTrainerInit:
    """Tests for GaussianTrainer initialization."""

    def create_synthetic_data(self, h=512, w=512):
        """Create synthetic test data."""
        rgb = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

        depth = np.full((h, w), 2.0, dtype=np.float32)
        depth += np.random.randn(h, w).astype(np.float32) * 0.1

        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        y, x = np.ogrid[:h, :w]
        mask = ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).astype(np.uint8)

        return rgb, depth, mask

    def test_trainer_init(self):
        """Test basic trainer initialization."""
        rgb, depth, mask = self.create_synthetic_data(256, 256)

        camera = CameraParams(
            fx=1000.0, fy=1000.0, cx=127.5, cy=127.5, width=256, height=256,
        )
        config = GaussianConfig(sh_degree=3)

        trainer = GaussianTrainer(
            rgb=rgb, depth=depth, mask=mask,
            camera_params=camera, config=config, device="cpu",
        )

        assert trainer.rgb.shape == (256, 256, 3)
        assert trainer.depth.shape == (256, 256)
        assert trainer.mask.shape == (256, 256)
        assert trainer.num_gaussians == 0

    def test_initialize_gaussians(self):
        """Test Gaussian initialization from depth map."""
        h, w = 512, 512
        rgb, depth, mask = self.create_synthetic_data(h, w)

        camera = CameraParams(
            fx=1000.0, fy=1000.0,
            cx=(w - 1) / 2.0, cy=(h - 1) / 2.0,
            width=w, height=h,
        )
        config = GaussianConfig(sh_degree=3, opacity_init=0.9)

        trainer = GaussianTrainer(
            rgb=rgb, depth=depth, mask=mask,
            camera_params=camera, config=config, device="cpu",
        )

        num_gaussians = trainer.initialize_gaussians()

        expected_points = mask.sum()
        assert num_gaussians == expected_points, (
            f"Expected {expected_points}, got {num_gaussians}"
        )

        assert trainer.means.shape == (num_gaussians, 3)
        assert trainer.scales.shape == (num_gaussians, 3)
        assert trainer.rotations.shape == (num_gaussians, 4)
        assert trainer.sh_coeffs.shape == (num_gaussians, 16, 3)
        assert trainer.opacities.shape == (num_gaussians,)

        z_values = trainer.means[:, 2].detach().numpy()
        assert z_values.min() >= 0.4, f"Z min {z_values.min()} too low"
        assert z_values.max() <= 2.6, f"Z max {z_values.max()} too high"

        assert torch.allclose(trainer.rotations[:, 0], torch.ones(num_gaussians))
        assert torch.allclose(trainer.rotations[:, 1:], torch.zeros(num_gaussians, 3))

        assert trainer.optimizer is not None

    def test_get_parameters(self):
        """Test get_parameters returns correct structure."""
        rgb, depth, mask = self.create_synthetic_data(128, 128)

        camera = CameraParams(
            fx=500.0, fy=500.0, cx=63.5, cy=63.5, width=128, height=128,
        )
        config = GaussianConfig(sh_degree=0)

        trainer = GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        params = trainer.get_parameters()

        assert "means" in params
        assert "scales" in params
        assert "rotations" in params
        assert "sh_coeffs" in params
        assert "opacities" in params

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
        from human3d.reconstruct.pointcloud import depth_to_pointcloud

        h, w = 64, 64
        depth_raw = np.random.rand(h, w).astype(np.float32) * 10
        bgr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.ones((h, w), dtype=np.uint8)

        fx, fy = 1000.0, 1000.0
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

        # Original implementation
        pcd = depth_to_pointcloud(depth_raw, bgr, fx, fy)
        xyz_original = np.asarray(pcd.points)

        # New implementation
        depth_normalized = gu.normalize_depth_to_metric(
            depth_raw, min_depth=0.5, max_depth=2.5
        )
        xyz_new = gu.depth_to_xyz(depth_normalized, mask, fx, fy, cx, cy)

        assert xyz_original.shape[0] == xyz_new.shape[0], (
            f"Point count mismatch: original {xyz_original.shape[0]} vs new {xyz_new.shape[0]}"
        )

        xyz_original_sorted = xyz_original[np.lexsort(xyz_original.T)]
        xyz_new_sorted = xyz_new[np.lexsort(xyz_new.T)]

        np.testing.assert_allclose(
            xyz_original_sorted, xyz_new_sorted,
            rtol=1e-4, atol=1e-6,
            err_msg="XYZ coordinates don't match original implementation",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
