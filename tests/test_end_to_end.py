"""
End-to-end tests for Gaussian Splatting pipeline.

Tests the complete workflow from image to PLY export.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytest
import yaml
import cv2

from human3d.reconstruct.gaussian_trainer import GaussianTrainer, GaussianConfig, CameraParams
from human3d.reconstruct import gaussian_utils
from human3d.export.ply_exporter import load_gaussian_ply


def load_fixtures():
    """Load test fixtures."""
    fixtures_dir = Path(__file__).parent / "fixtures"

    # Check if fixtures exist, generate if not
    if not (fixtures_dir / "person.jpg").exists():
        import subprocess

        subprocess.run([sys.executable, str(fixtures_dir / "generate_fixtures.py")])

    # Load RGB image
    rgb_path = fixtures_dir / "person.jpg"
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Load depth
    depth = np.load(str(fixtures_dir / "person_depth.npy"))

    # Load mask
    mask = np.load(str(fixtures_dir / "person_mask.npy"))

    # Load camera params
    with open(fixtures_dir / "camera_params.yaml", "r") as f:
        camera_params = yaml.safe_load(f)

    return rgb, depth, mask, camera_params


# ==============================================================================
# PSNR Helper
# ==============================================================================


def compute_psnr(img1, img2, mask=None):
    """Compute Peak Signal-to-Noise Ratio between two images."""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    img1 = np.clip(img1.astype(np.float32), 0, 1)
    img2 = np.clip(img2.astype(np.float32), 0, 1)

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = mask.astype(bool)

        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]

        diff = (img1 - img2) ** 2
        mse = diff[np.broadcast_to(mask, img1.shape)].mean()
    else:
        mse = ((img1 - img2) ** 2).mean()

    if mse < 1e-10:
        return 100.0

    return 10 * np.log10(1.0 / mse)


# ==============================================================================
# End-to-End Tests
# ==============================================================================


class TestGaussianPipeline:
    """Tests for the complete Gaussian splatting pipeline."""

    @pytest.fixture
    def fixture_data(self):
        return load_fixtures()

    def test_gaussian_pipeline_basic(self, fixture_data):
        """Test full pipeline from image to PLY (fast test)."""
        rgb, depth, mask, camera_params = fixture_data

        camera = CameraParams(
            fx=camera_params["fx"], fy=camera_params["fy"],
            cx=camera_params["cx"], cy=camera_params["cy"],
            width=camera_params["width"], height=camera_params["height"],
        )

        config = GaussianConfig(sh_degree=0, num_iterations=20)

        trainer = GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        num_gaussians = trainer.initialize_gaussians()

        assert num_gaussians > 0, "Should initialize Gaussians"

        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.optimize(
                num_iterations=20, log_every=100, save_every=0, output_dir=tmpdir,
            )

            assert len(history["loss"]) == 20

            ply_path = Path(tmpdir) / "output.ply"
            trainer.export_ply(str(ply_path))

            assert ply_path.exists()
            assert ply_path.stat().st_size > 0

            data = load_gaussian_ply(ply_path)
            assert data["num_gaussians"] == num_gaussians

    def test_loss_decreases_during_training(self, fixture_data):
        """Test that loss decreases during training."""
        rgb, depth, mask, camera_params = fixture_data

        camera = CameraParams(
            fx=camera_params["fx"], fy=camera_params["fy"],
            cx=camera_params["cx"], cy=camera_params["cy"],
            width=camera_params["width"], height=camera_params["height"],
        )

        config = GaussianConfig(sh_degree=0)

        trainer = GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.optimize(
                num_iterations=30, log_every=100, save_every=0, output_dir=tmpdir,
            )

        initial_loss = np.mean(history["loss"][:5])
        final_loss = np.mean(history["loss"][-5:])

        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )


class TestInitializationConsistency:
    """Tests for initialization consistency."""

    @pytest.fixture
    def fixture_data(self):
        return load_fixtures()

    def test_initialization_produces_valid_points(self, fixture_data):
        """Verify Gaussian initialization produces valid 3D points."""
        rgb, depth, mask, camera_params = fixture_data

        camera = CameraParams(
            fx=camera_params["fx"], fy=camera_params["fy"],
            cx=camera_params["cx"], cy=camera_params["cy"],
            width=camera_params["width"], height=camera_params["height"],
        )

        config = GaussianConfig(sh_degree=0, position_noise=0.0)

        trainer = GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        means = trainer.means.detach().cpu().numpy()

        assert np.all(np.isfinite(means)), "All points should be finite"
        assert means.min() > -10, "Points shouldn't be too far negative"
        assert means.max() < 10, "Points shouldn't be too far positive"

    def test_initialization_matches_depth_to_xyz(self, fixture_data):
        """Verify Gaussian init uses correct depth unprojection."""
        rgb, depth, mask, camera_params = fixture_data

        depth_tensor = torch.from_numpy(depth.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        depth_normalized = gaussian_utils.normalize_depth_to_metric(depth_tensor, 0.5, 2.5)

        xyz_direct = gaussian_utils.depth_to_xyz(
            depth_normalized, mask_tensor,
            fx=camera_params["fx"], fy=camera_params["fy"],
            cx=camera_params["cx"], cy=camera_params["cy"],
        )

        camera = CameraParams(
            fx=camera_params["fx"], fy=camera_params["fy"],
            cx=camera_params["cx"], cy=camera_params["cy"],
            width=camera_params["width"], height=camera_params["height"],
        )

        config = GaussianConfig(sh_degree=0, position_noise=0.0)

        trainer = GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        xyz_trainer = trainer.means.detach()

        assert xyz_direct.shape == xyz_trainer.shape, (
            f"Shape mismatch: {xyz_direct.shape} vs {xyz_trainer.shape}"
        )

        np.testing.assert_allclose(
            xyz_direct.numpy(), xyz_trainer.numpy(), rtol=1e-5, atol=1e-6,
        )


class TestRenderingQuality:
    """Tests for rendering quality."""

    @pytest.fixture
    def fixture_data(self):
        return load_fixtures()

    def test_initial_render_produces_valid_image(self, fixture_data):
        """Test that initial render produces a valid image."""
        rgb, depth, mask, camera_params = fixture_data

        camera = CameraParams(
            fx=camera_params["fx"], fy=camera_params["fy"],
            cx=camera_params["cx"], cy=camera_params["cy"],
            width=camera_params["width"], height=camera_params["height"],
        )

        config = GaussianConfig(sh_degree=0)

        trainer = GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        rendered_rgb, rendered_depth, rendered_alpha = trainer.render_view()

        H, W = camera_params["height"], camera_params["width"]
        assert rendered_rgb.shape == (H, W, 3), f"RGB shape: {rendered_rgb.shape}"
        assert rendered_depth.shape == (H, W), f"Depth shape: {rendered_depth.shape}"
        assert rendered_alpha.shape == (H, W), f"Alpha shape: {rendered_alpha.shape}"

        assert rendered_rgb.min() >= 0, "RGB should be >= 0"
        assert rendered_rgb.max() <= 1, "RGB should be <= 1"
        assert rendered_alpha.min() >= 0, "Alpha should be >= 0"
        assert rendered_alpha.max() <= 1, "Alpha should be <= 1"

    @pytest.mark.slow
    def test_rendering_quality_after_training(self, fixture_data):
        """Test that rendering quality improves after training (PSNR > 15 dB)."""
        rgb, depth, mask, camera_params = fixture_data

        camera = CameraParams(
            fx=camera_params["fx"], fy=camera_params["fy"],
            cx=camera_params["cx"], cy=camera_params["cy"],
            width=camera_params["width"], height=camera_params["height"],
        )

        config = GaussianConfig(
            sh_degree=0, loss_weight_l1=0.8, loss_weight_ssim=0.2,
        )

        trainer = GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.optimize(
                num_iterations=500, log_every=100, save_every=0, output_dir=tmpdir,
            )

        rendered_rgb, _, _ = trainer.render_view()

        target_rgb = rgb.astype(np.float32) / 255.0
        psnr = compute_psnr(rendered_rgb, target_rgb, mask)

        print(f"\nFinal PSNR: {psnr:.2f} dB")
        assert psnr > 15, f"PSNR should be > 15 dB, got {psnr:.2f} dB"


class TestCUDAPipeline:
    """Tests for CUDA pipeline (skipped if CUDA not available)."""

    @pytest.fixture
    def fixture_data(self):
        return load_fixtures()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_pipeline(self, fixture_data):
        """Test pipeline on CUDA."""
        rgb, depth, mask, camera_params = fixture_data

        camera = CameraParams(
            fx=camera_params["fx"], fy=camera_params["fy"],
            cx=camera_params["cx"], cy=camera_params["cy"],
            width=camera_params["width"], height=camera_params["height"],
        )

        config = GaussianConfig(sh_degree=0)

        trainer = GaussianTrainer(rgb, depth, mask, camera, config, device="cuda")
        trainer.initialize_gaussians()

        assert trainer.means.device.type == "cuda"

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.optimize(
                num_iterations=50, log_every=100, save_every=0, output_dir=tmpdir,
            )

            ply_path = Path(tmpdir) / "cuda_output.ply"
            trainer.export_ply(str(ply_path))

            assert ply_path.exists()
            data = load_gaussian_ply(ply_path)
            assert data["num_gaussians"] == trainer.num_gaussians


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
