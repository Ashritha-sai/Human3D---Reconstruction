"""
Test script for PLY exporter.

Tests that:
1. PLY files are saved correctly
2. PLY files can be loaded back
3. Data roundtrip preserves values
4. File sizes are reasonable
"""

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def _import_exporter():
    """Import ply_exporter module directly to avoid package dependencies."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ply_exporter",
        Path(__file__).parent.parent / "src" / "human3d" / "export" / "ply_exporter.py"
    )
    ply_exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ply_exporter)

    return {
        'save': ply_exporter.save_gaussian_ply,
        'load': ply_exporter.load_gaussian_ply,
        'validate': ply_exporter.validate_gaussian_attributes,
        'header': ply_exporter.construct_ply_header,
        'sh_count': ply_exporter.get_sh_coefficient_count,
    }


def create_test_gaussians(n=1000, sh_degree=0):
    """Create test Gaussian data."""
    np.random.seed(42)

    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.randn(n, 3).astype(np.float32) * 0.5 - 2.0  # log-space

    rotations = np.random.randn(n, 4).astype(np.float32)
    rotations = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-8)

    num_sh = (sh_degree + 1) ** 2
    sh_coeffs = np.random.randn(n, num_sh, 3).astype(np.float32) * 0.5

    opacities = np.random.rand(n).astype(np.float32)

    return means, scales, rotations, sh_coeffs, opacities


# ==============================================================================
# Tests
# ==============================================================================

class TestSHCoefficientCount:
    """Tests for SH coefficient counting."""

    def test_degree_0(self):
        exp = _import_exporter()
        assert exp['sh_count'](0) == 1

    def test_degree_1(self):
        exp = _import_exporter()
        assert exp['sh_count'](1) == 4

    def test_degree_2(self):
        exp = _import_exporter()
        assert exp['sh_count'](2) == 9

    def test_degree_3(self):
        exp = _import_exporter()
        assert exp['sh_count'](3) == 16


class TestValidation:
    """Tests for input validation."""

    def test_valid_input(self):
        exp = _import_exporter()
        means, scales, rotations, sh_coeffs, opacities = create_test_gaussians(100)
        n = exp['validate'](means, scales, rotations, sh_coeffs, opacities)
        assert n == 100

    def test_invalid_means_shape(self):
        exp = _import_exporter()
        means = np.random.randn(100, 2).astype(np.float32)  # Wrong shape
        _, scales, rotations, sh_coeffs, opacities = create_test_gaussians(100)
        with pytest.raises(ValueError, match="means must have shape"):
            exp['validate'](means, scales, rotations, sh_coeffs, opacities)

    def test_mismatched_counts(self):
        exp = _import_exporter()
        means, scales, rotations, sh_coeffs, opacities = create_test_gaussians(100)
        scales = scales[:50]  # Different count
        with pytest.raises(ValueError, match="scales has"):
            exp['validate'](means, scales, rotations, sh_coeffs, opacities)


class TestPLYHeader:
    """Tests for PLY header construction."""

    def test_header_format(self):
        exp = _import_exporter()
        header = exp['header'](1000, sh_degree=0)

        assert header.startswith("ply\n")
        assert "format binary_little_endian" in header
        assert "element vertex 1000" in header
        assert "property float x" in header
        assert "property float f_dc_0" in header
        assert "property float opacity" in header
        assert "end_header" in header

    def test_header_sh_degree_3(self):
        exp = _import_exporter()
        header = exp['header'](500, sh_degree=3)

        # Should have f_rest_0 through f_rest_44 (15 coeffs * 3 channels - 1)
        assert "f_rest_0" in header
        assert "f_rest_44" in header


class TestSaveLoad:
    """Tests for save and load functionality."""

    def test_save_creates_file(self):
        exp = _import_exporter()
        means, scales, rotations, sh_coeffs, opacities = create_test_gaussians(100)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.ply"
            exp['save'](means, scales, rotations, sh_coeffs, opacities, filepath)

            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_roundtrip_degree_0(self):
        exp = _import_exporter()
        means, scales, rotations, sh_coeffs, opacities = create_test_gaussians(100, sh_degree=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.ply"
            exp['save'](means, scales, rotations, sh_coeffs, opacities, filepath)

            data = exp['load'](filepath)

            assert data['num_gaussians'] == 100
            np.testing.assert_allclose(data['means'], means, rtol=1e-5)
            np.testing.assert_allclose(data['scales'], scales, rtol=1e-5)
            np.testing.assert_allclose(data['opacities'], opacities, rtol=1e-5)

    def test_roundtrip_degree_3(self):
        exp = _import_exporter()
        means, scales, rotations, sh_coeffs, opacities = create_test_gaussians(50, sh_degree=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sh3.ply"
            exp['save'](means, scales, rotations, sh_coeffs, opacities, filepath)

            data = exp['load'](filepath)

            assert data['num_gaussians'] == 50
            assert data['sh_coeffs'].shape == (50, 16, 3)
            np.testing.assert_allclose(data['sh_coeffs'], sh_coeffs, rtol=1e-5)

    def test_file_size_reasonable(self):
        """Test that file size is proportional to Gaussian count."""
        exp = _import_exporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Small file
            means1, scales1, rots1, sh1, ops1 = create_test_gaussians(100)
            path1 = Path(tmpdir) / "small.ply"
            exp['save'](means1, scales1, rots1, sh1, ops1, path1)
            size1 = path1.stat().st_size

            # Larger file
            means2, scales2, rots2, sh2, ops2 = create_test_gaussians(1000)
            path2 = Path(tmpdir) / "large.ply"
            exp['save'](means2, scales2, rots2, sh2, ops2, path2)
            size2 = path2.stat().st_size

            # Size should scale roughly linearly
            ratio = size2 / size1
            assert 8 < ratio < 12, f"File size ratio {ratio} is unexpected"


class TestTrainerExport:
    """Test GaussianTrainer.export_ply integration."""

    def test_trainer_export(self):
        """Test exporting from a trained model."""
        import importlib.util
        import types

        # Import gaussian_utils
        spec_utils = importlib.util.spec_from_file_location(
            "gaussian_utils",
            Path(__file__).parent.parent / "src" / "human3d" / "reconstruct" / "gaussian_utils.py"
        )
        gaussian_utils = importlib.util.module_from_spec(spec_utils)
        spec_utils.loader.exec_module(gaussian_utils)

        # Set up module hierarchy
        if 'human3d' not in sys.modules:
            sys.modules['human3d'] = types.ModuleType('human3d')
        if 'human3d.reconstruct' not in sys.modules:
            sys.modules['human3d.reconstruct'] = types.ModuleType('human3d.reconstruct')
            sys.modules['human3d'].reconstruct = sys.modules['human3d.reconstruct']
        if 'human3d.export' not in sys.modules:
            sys.modules['human3d.export'] = types.ModuleType('human3d.export')
            sys.modules['human3d'].export = sys.modules['human3d.export']

        sys.modules['human3d.reconstruct.gaussian_utils'] = gaussian_utils
        sys.modules['human3d.reconstruct'].gaussian_utils = gaussian_utils

        # Import losses
        spec_losses = importlib.util.spec_from_file_location(
            "human3d.reconstruct.losses",
            Path(__file__).parent.parent / "src" / "human3d" / "reconstruct" / "losses.py"
        )
        losses = importlib.util.module_from_spec(spec_losses)
        sys.modules['human3d.reconstruct.losses'] = losses
        sys.modules['human3d.reconstruct'].losses = losses
        spec_losses.loader.exec_module(losses)

        # Import ply_exporter
        spec_ply = importlib.util.spec_from_file_location(
            "human3d.export.ply_exporter",
            Path(__file__).parent.parent / "src" / "human3d" / "export" / "ply_exporter.py"
        )
        ply_exporter = importlib.util.module_from_spec(spec_ply)
        sys.modules['human3d.export.ply_exporter'] = ply_exporter
        sys.modules['human3d.export'].ply_exporter = ply_exporter
        spec_ply.loader.exec_module(ply_exporter)

        # Import gaussian_trainer
        spec_trainer = importlib.util.spec_from_file_location(
            "human3d.reconstruct.gaussian_trainer",
            Path(__file__).parent.parent / "src" / "human3d" / "reconstruct" / "gaussian_trainer.py"
        )
        gaussian_trainer = importlib.util.module_from_spec(spec_trainer)
        sys.modules['human3d.reconstruct.gaussian_trainer'] = gaussian_trainer
        spec_trainer.loader.exec_module(gaussian_trainer)

        # Create test data
        H, W = 32, 32
        rgb = np.random.rand(H, W, 3).astype(np.float32)
        depth = np.full((H, W), 2.0, dtype=np.float32)
        mask = np.ones((H, W), dtype=np.uint8)

        camera = gaussian_trainer.CameraParams(
            fx=500.0, fy=500.0,
            cx=15.5, cy=15.5,
            width=W, height=H,
        )
        config = gaussian_trainer.GaussianConfig(sh_degree=0)

        trainer = gaussian_trainer.GaussianTrainer(rgb, depth, mask, camera, config, device="cpu")
        trainer.initialize_gaussians()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "exported.ply"
            trainer.export_ply(str(output_path))

            assert output_path.exists()
            size_bytes = output_path.stat().st_size
            print(f"\nExported PLY size: {size_bytes / 1024:.1f} KB")
            print(f"Gaussians: {trainer.num_gaussians}")
            print(f"Bytes per Gaussian: {size_bytes / trainer.num_gaussians:.1f}")

            # Load and verify
            data = ply_exporter.load_gaussian_ply(output_path)
            assert data['num_gaussians'] == trainer.num_gaussians


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
