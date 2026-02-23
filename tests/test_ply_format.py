"""
Tests for PLY format validation.

Verifies that exported PLY files have correct structure and valid values.
"""

import sys
from pathlib import Path
import importlib.util
import tempfile

import numpy as np
import pytest

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def _import_exporter():
    """Import ply_exporter module directly."""
    spec = importlib.util.spec_from_file_location(
        "ply_exporter",
        Path(__file__).parent.parent / "src" / "human3d" / "export" / "ply_exporter.py",
    )
    ply_exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ply_exporter)
    return ply_exporter


def create_test_gaussians(n=100, sh_degree=0):
    """Create test Gaussian data."""
    np.random.seed(42)

    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.randn(n, 3).astype(np.float32) * 0.5 - 2.0

    rotations = np.random.randn(n, 4).astype(np.float32)
    rotations = rotations / (np.linalg.norm(rotations, axis=1, keepdims=True) + 1e-8)

    num_sh = (sh_degree + 1) ** 2
    sh_coeffs = np.random.randn(n, num_sh, 3).astype(np.float32) * 0.5

    opacities = np.random.rand(n).astype(np.float32)

    return means, scales, rotations, sh_coeffs, opacities


# ==============================================================================
# PLY Format Tests
# ==============================================================================


class TestPLYFieldsExist:
    """Verify all required PLY fields exist."""

    def test_position_fields(self):
        """Test that position fields (x, y, z) exist."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(50)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            # Check position fields
            assert "x" in vertex.data.dtype.names
            assert "y" in vertex.data.dtype.names
            assert "z" in vertex.data.dtype.names

    def test_normal_fields(self):
        """Test that normal fields (nx, ny, nz) exist."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(50)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            assert "nx" in vertex.data.dtype.names
            assert "ny" in vertex.data.dtype.names
            assert "nz" in vertex.data.dtype.names

    def test_dc_color_fields(self):
        """Test that DC color fields exist."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(50)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            assert "f_dc_0" in vertex.data.dtype.names
            assert "f_dc_1" in vertex.data.dtype.names
            assert "f_dc_2" in vertex.data.dtype.names

    def test_opacity_field(self):
        """Test that opacity field exists."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(50)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            assert "opacity" in vertex.data.dtype.names

    def test_scale_fields(self):
        """Test that scale fields exist."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(50)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            assert "scale_0" in vertex.data.dtype.names
            assert "scale_1" in vertex.data.dtype.names
            assert "scale_2" in vertex.data.dtype.names

    def test_rotation_fields(self):
        """Test that rotation quaternion fields exist."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(50)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            assert "rot_0" in vertex.data.dtype.names
            assert "rot_1" in vertex.data.dtype.names
            assert "rot_2" in vertex.data.dtype.names
            assert "rot_3" in vertex.data.dtype.names

    def test_sh_rest_fields_degree_3(self):
        """Test that SH rest fields exist for degree 3."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(50, sh_degree=3)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            # For degree 3: 16 coeffs, 15 rest * 3 channels = 45 f_rest fields
            for i in range(45):
                assert f"f_rest_{i}" in vertex.data.dtype.names, f"Missing f_rest_{i}"


class TestPLYValueRanges:
    """Verify PLY values are in valid ranges."""

    def test_positions_finite(self):
        """Test that positions are finite."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(100)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            x = np.array(vertex["x"])
            y = np.array(vertex["y"])
            z = np.array(vertex["z"])

            assert np.all(np.isfinite(x)), "x should be finite"
            assert np.all(np.isfinite(y)), "y should be finite"
            assert np.all(np.isfinite(z)), "z should be finite"

    def test_opacity_range(self):
        """Test that opacity is in valid range [0, 1]."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(100)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            opacity = np.array(vertex["opacity"])

            assert np.all(opacity >= 0), f"Opacity min: {opacity.min()}"
            assert np.all(opacity <= 1), f"Opacity max: {opacity.max()}"

    def test_quaternions_normalized(self):
        """Test that quaternions are normalized (magnitude ~1)."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(100)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            rot = np.stack(
                [
                    vertex["rot_0"],
                    vertex["rot_1"],
                    vertex["rot_2"],
                    vertex["rot_3"],
                ],
                axis=1,
            )

            norms = np.linalg.norm(rot, axis=1)

            # Should be close to 1
            np.testing.assert_allclose(norms, 1.0, rtol=1e-5, atol=1e-5)

    def test_scales_reasonable(self):
        """Test that scales are in reasonable range."""
        from plyfile import PlyData

        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(100)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            vertex = data["vertex"]

            scale_0 = np.array(vertex["scale_0"])
            scale_1 = np.array(vertex["scale_1"])
            scale_2 = np.array(vertex["scale_2"])

            # Log-space scales should typically be in [-10, 5]
            all_scales = np.concatenate([scale_0, scale_1, scale_2])

            assert np.all(np.isfinite(all_scales)), "Scales should be finite"
            assert all_scales.min() > -20, f"Scales too small: {all_scales.min()}"
            assert all_scales.max() < 10, f"Scales too large: {all_scales.max()}"


class TestPLYBinaryFormat:
    """Verify PLY binary format is correct."""

    def test_is_binary_format(self):
        """Test that PLY is saved in binary format."""
        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(100)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            # Read first few bytes
            with open(f.name, "rb") as fp:
                header = fp.read(200).decode("ascii", errors="ignore")

            assert "binary_little_endian" in header, "Should be binary format"

    def test_vertex_count_matches(self):
        """Test that vertex count matches input."""
        from plyfile import PlyData

        ply = _import_exporter()
        n = 73  # Arbitrary number
        means, scales, rots, sh, ops = create_test_gaussians(n)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)

            data = PlyData.read(f.name)
            assert len(data["vertex"]) == n


class TestPLYRoundtrip:
    """Test save/load roundtrip preserves data."""

    def test_roundtrip_positions(self):
        """Test positions survive roundtrip."""
        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(100)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)
            data = ply.load_gaussian_ply(f.name)

        np.testing.assert_allclose(data["means"], means, rtol=1e-5)

    def test_roundtrip_scales(self):
        """Test scales survive roundtrip."""
        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(100)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)
            data = ply.load_gaussian_ply(f.name)

        np.testing.assert_allclose(data["scales"], scales, rtol=1e-5)

    def test_roundtrip_rotations(self):
        """Test rotations survive roundtrip."""
        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(100)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)
            data = ply.load_gaussian_ply(f.name)

        np.testing.assert_allclose(data["rotations"], rots, rtol=1e-5)

    def test_roundtrip_sh_coeffs(self):
        """Test SH coefficients survive roundtrip."""
        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(100, sh_degree=3)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)
            data = ply.load_gaussian_ply(f.name)

        np.testing.assert_allclose(data["sh_coeffs"], sh, rtol=1e-5)

    def test_roundtrip_opacities(self):
        """Test opacities survive roundtrip."""
        ply = _import_exporter()
        means, scales, rots, sh, ops = create_test_gaussians(100)

        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
            ply.save_gaussian_ply(means, scales, rots, sh, ops, f.name)
            data = ply.load_gaussian_ply(f.name)

        np.testing.assert_allclose(data["opacities"], ops, rtol=1e-5)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
