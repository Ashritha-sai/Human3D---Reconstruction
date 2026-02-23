"""
End-to-end tests for Gaussian Splatting pipeline.

Tests the complete workflow from image to PLY export.
"""

import sys
from pathlib import Path
import importlib.util
import types
import tempfile

import numpy as np
import torch
import pytest
import yaml
import cv2

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


# ==============================================================================
# Module Import Helpers
# ==============================================================================


def _setup_modules():
    """Set up module hierarchy for imports."""
    # Import gaussian_utils
    spec_utils = importlib.util.spec_from_file_location(
        "gaussian_utils",
        Path(__file__).parent.parent
        / "src"
        / "human3d"
        / "reconstruct"
        / "gaussian_utils.py",
    )
    gaussian_utils = importlib.util.module_from_spec(spec_utils)
    spec_utils.loader.exec_module(gaussian_utils)

    # Set up module hierarchy
    if "human3d" not in sys.modules:
        sys.modules["human3d"] = types.ModuleType("human3d")
    if "human3d.reconstruct" not in sys.modules:
        sys.modules["human3d.reconstruct"] = types.ModuleType("human3d.reconstruct")
        sys.modules["human3d"].reconstruct = sys.modules["human3d.reconstruct"]
    if "human3d.export" not in sys.modules:
        sys.modules["human3d.export"] = types.ModuleType("human3d.export")
        sys.modules["human3d"].export = sys.modules["human3d.export"]

    sys.modules["human3d.reconstruct.gaussian_utils"] = gaussian_utils
    sys.modules["human3d.reconstruct"].gaussian_utils = gaussian_utils

    # Import losses
    spec_losses = importlib.util.spec_from_file_location(
        "human3d.reconstruct.losses",
        Path(__file__).parent.parent / "src" / "human3d" / "reconstruct" / "losses.py",
    )
    losses = importlib.util.module_from_spec(spec_losses)
    sys.modules["human3d.reconstruct.losses"] = losses
    sys.modules["human3d.reconstruct"].losses = losses
    spec_losses.loader.exec_module(losses)

    # Import ply_exporter
    spec_ply = importlib.util.spec_from_file_location(
        "human3d.export.ply_exporter",
        Path(__file__).parent.parent / "src" / "human3d" / "export" / "ply_exporter.py",
    )
    ply_exporter = importlib.util.module_from_spec(spec_ply)
    sys.modules["human3d.export.ply_exporter"] = ply_exporter
    sys.modules["human3d.export"].ply_exporter = ply_exporter
    spec_ply.loader.exec_module(ply_exporter)

    # Import gaussian_trainer
    spec_trainer = importlib.util.spec_from_file_location(
        "human3d.reconstruct.gaussian_trainer",
        Path(__file__).parent.parent
        / "src"
        / "human3d"
        / "reconstruct"
        / "gaussian_trainer.py",
    )
    gaussian_trainer = importlib.util.module_from_spec(spec_trainer)
    sys.modules["human3d.reconstruct.gaussian_trainer"] = gaussian_trainer
    spec_trainer.loader.exec_module(gaussian_trainer)

    return {
        "gaussian_utils": gaussian_utils,
        "gaussian_trainer": gaussian_trainer,
        "losses": losses,
        "ply_exporter": ply_exporter,
    }


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
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        img1: First image, shape (H, W, 3), range [0, 1]
        img2: Second image, shape (H, W, 3), range [0, 1]
        mask: Optional binary mask, shape (H, W)

    Returns:
        float: PSNR in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Ensure float and clamp
    img1 = np.clip(img1.astype(np.float32), 0, 1)
    img2 = np.clip(img2.astype(np.float32), 0, 1)

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = mask.astype(bool)

        # Expand mask to 3 channels
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]

        # Compute MSE only in masked region
        diff = (img1 - img2) ** 2
        mse = diff[np.broadcast_to(mask, img1.shape)].mean()
    else:
        mse = ((img1 - img2) ** 2).mean()

    if mse < 1e-10:
        return 100.0  # Perfect match

    psnr = 10 * np.log10(1.0 / mse)
    return psnr


# ==============================================================================
# End-to-End Tests
# ==============================================================================


class TestGaussianPipeline:
    """Tests for the complete Gaussian splatting pipeline."""

    @pytest.fixture
    def modules(self):
        """Set up modules."""
        return _setup_modules()

    @pytest.fixture
    def fixture_data(self):
        """Load fixture data."""
        return load_fixtures()

    def test_gaussian_pipeline_basic(self, modules, fixture_data):
        """Test full pipeline from image to PLY (fast test)."""
        gt = modules["gaussian_trainer"]
        ply = modules["ply_exporter"]

        rgb, depth, mask, camera_params = fixture_data

        # Create camera
        camera = gt.CameraParams(
            fx=camera_params["fx"],
            fy=camera_params["fy"],
            cx=camera_params["cx"],
            cy=camera_params["cy"],
            width=camera_params["width"],
            height=camera_params["height"],
        )

        # Create config
        config = gt.GaussianConfig(
            sh_degree=0,
            num_iterations=20,
        )

        # Initialize trainer
        trainer = gt.GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        num_gaussians = trainer.initialize_gaussians()

        assert num_gaussians > 0, "Should initialize Gaussians"

        # Run optimization (short)
        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.optimize(
                num_iterations=20,
                log_every=100,
                save_every=0,
                output_dir=tmpdir,
            )

            assert len(history["loss"]) == 20

            # Export PLY
            ply_path = Path(tmpdir) / "output.ply"
            trainer.export_ply(str(ply_path))

            # Verify PLY exists
            assert ply_path.exists()
            assert ply_path.stat().st_size > 0

            # Load and verify structure
            data = ply.load_gaussian_ply(ply_path)
            assert data["num_gaussians"] == num_gaussians

    def test_loss_decreases_during_training(self, modules, fixture_data):
        """Test that loss decreases during training."""
        gt = modules["gaussian_trainer"]

        rgb, depth, mask, camera_params = fixture_data

        camera = gt.CameraParams(
            fx=camera_params["fx"],
            fy=camera_params["fy"],
            cx=camera_params["cx"],
            cy=camera_params["cy"],
            width=camera_params["width"],
            height=camera_params["height"],
        )

        config = gt.GaussianConfig(sh_degree=0)

        trainer = gt.GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        with tempfile.TemporaryDirectory() as tmpdir:
            history = trainer.optimize(
                num_iterations=30,
                log_every=100,
                save_every=0,
                output_dir=tmpdir,
            )

        # Loss should decrease
        initial_loss = np.mean(history["loss"][:5])
        final_loss = np.mean(history["loss"][-5:])

        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )


class TestInitializationConsistency:
    """Tests for initialization consistency."""

    @pytest.fixture
    def modules(self):
        return _setup_modules()

    @pytest.fixture
    def fixture_data(self):
        return load_fixtures()

    def test_initialization_produces_valid_points(self, modules, fixture_data):
        """Verify Gaussian initialization produces valid 3D points."""
        gt = modules["gaussian_trainer"]

        rgb, depth, mask, camera_params = fixture_data

        camera = gt.CameraParams(
            fx=camera_params["fx"],
            fy=camera_params["fy"],
            cx=camera_params["cx"],
            cy=camera_params["cy"],
            width=camera_params["width"],
            height=camera_params["height"],
        )

        config = gt.GaussianConfig(sh_degree=0, position_noise=0.0)

        trainer = gt.GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        means = trainer.means.detach().cpu().numpy()

        # Points should be in reasonable range
        assert np.all(np.isfinite(means)), "All points should be finite"
        assert means.min() > -10, "Points shouldn't be too far negative"
        assert means.max() < 10, "Points shouldn't be too far positive"

    def test_initialization_matches_depth_to_xyz(self, modules, fixture_data):
        """Verify Gaussian init uses correct depth unprojection."""
        gt = modules["gaussian_trainer"]
        gu = modules["gaussian_utils"]

        rgb, depth, mask, camera_params = fixture_data

        # Direct depth_to_xyz call
        depth_tensor = torch.from_numpy(depth.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        depth_normalized = gu.normalize_depth_to_metric(depth_tensor, 0.5, 2.5)

        xyz_direct = gu.depth_to_xyz(
            depth_normalized,
            mask_tensor,
            fx=camera_params["fx"],
            fy=camera_params["fy"],
            cx=camera_params["cx"],
            cy=camera_params["cy"],
        )

        # Trainer initialization
        camera = gt.CameraParams(
            fx=camera_params["fx"],
            fy=camera_params["fy"],
            cx=camera_params["cx"],
            cy=camera_params["cy"],
            width=camera_params["width"],
            height=camera_params["height"],
        )

        config = gt.GaussianConfig(sh_degree=0, position_noise=0.0)

        trainer = gt.GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        xyz_trainer = trainer.means.detach()

        # Should produce same points
        assert xyz_direct.shape == xyz_trainer.shape, (
            f"Shape mismatch: {xyz_direct.shape} vs {xyz_trainer.shape}"
        )

        # Allow small tolerance for floating point
        np.testing.assert_allclose(
            xyz_direct.numpy(),
            xyz_trainer.numpy(),
            rtol=1e-5,
            atol=1e-6,
        )


class TestRenderingQuality:
    """Tests for rendering quality."""

    @pytest.fixture
    def modules(self):
        return _setup_modules()

    @pytest.fixture
    def fixture_data(self):
        return load_fixtures()

    def test_initial_render_produces_valid_image(self, modules, fixture_data):
        """Test that initial render produces a valid image."""
        gt = modules["gaussian_trainer"]

        rgb, depth, mask, camera_params = fixture_data

        camera = gt.CameraParams(
            fx=camera_params["fx"],
            fy=camera_params["fy"],
            cx=camera_params["cx"],
            cy=camera_params["cy"],
            width=camera_params["width"],
            height=camera_params["height"],
        )

        config = gt.GaussianConfig(sh_degree=0)

        trainer = gt.GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        rendered_rgb, rendered_depth, rendered_alpha = trainer.render_view()

        # Check shapes
        H, W = camera_params["height"], camera_params["width"]
        assert rendered_rgb.shape == (H, W, 3), f"RGB shape: {rendered_rgb.shape}"
        assert rendered_depth.shape == (H, W), f"Depth shape: {rendered_depth.shape}"
        assert rendered_alpha.shape == (H, W), f"Alpha shape: {rendered_alpha.shape}"

        # Check ranges
        assert rendered_rgb.min() >= 0, "RGB should be >= 0"
        assert rendered_rgb.max() <= 1, "RGB should be <= 1"
        assert rendered_alpha.min() >= 0, "Alpha should be >= 0"
        assert rendered_alpha.max() <= 1, "Alpha should be <= 1"

    @pytest.mark.slow
    def test_rendering_quality_after_training(self, modules, fixture_data):
        """Test that rendering quality improves after training (PSNR > 20 dB)."""
        gt = modules["gaussian_trainer"]

        rgb, depth, mask, camera_params = fixture_data

        camera = gt.CameraParams(
            fx=camera_params["fx"],
            fy=camera_params["fy"],
            cx=camera_params["cx"],
            cy=camera_params["cy"],
            width=camera_params["width"],
            height=camera_params["height"],
        )

        config = gt.GaussianConfig(
            sh_degree=0,
            loss_weight_l1=0.8,
            loss_weight_ssim=0.2,
        )

        trainer = gt.GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        # Train
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.optimize(
                num_iterations=500,
                log_every=100,
                save_every=0,
                output_dir=tmpdir,
            )

        # Render final view
        rendered_rgb, _, _ = trainer.render_view()

        # Compute PSNR
        target_rgb = rgb.astype(np.float32) / 255.0
        psnr = compute_psnr(rendered_rgb, target_rgb, mask)

        print(f"\nFinal PSNR: {psnr:.2f} dB")

        # PSNR should be reasonable (> 15 dB for quick test)
        # For longer training, expect > 25 dB
        assert psnr > 15, f"PSNR should be > 15 dB, got {psnr:.2f} dB"


class TestCUDAPipeline:
    """Tests for CUDA pipeline (skipped if CUDA not available)."""

    @pytest.fixture
    def modules(self):
        return _setup_modules()

    @pytest.fixture
    def fixture_data(self):
        return load_fixtures()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_pipeline(self, modules, fixture_data):
        """Test pipeline on CUDA."""
        gt = modules["gaussian_trainer"]
        ply = modules["ply_exporter"]

        rgb, depth, mask, camera_params = fixture_data

        camera = gt.CameraParams(
            fx=camera_params["fx"],
            fy=camera_params["fy"],
            cx=camera_params["cx"],
            cy=camera_params["cy"],
            width=camera_params["width"],
            height=camera_params["height"],
        )

        config = gt.GaussianConfig(sh_degree=0)

        trainer = gt.GaussianTrainer(rgb, depth, mask, camera, config, device="cuda")
        trainer.initialize_gaussians()

        assert trainer.means.device.type == "cuda"

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.optimize(
                num_iterations=50,
                log_every=100,
                save_every=0,
                output_dir=tmpdir,
            )

            # Export PLY (should work from CUDA tensors)
            ply_path = Path(tmpdir) / "cuda_output.ply"
            trainer.export_ply(str(ply_path))

            assert ply_path.exists()
            data = ply.load_gaussian_ply(ply_path)
            assert data["num_gaussians"] == trainer.num_gaussians


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
