"""Tests for point cloud reconstruction utilities."""

import numpy as np

from human3d.reconstruct.pointcloud import depth_to_pointcloud, save_ply


class TestDepthToPointcloud:
    """Test suite for depth_to_pointcloud function."""

    def test_basic_conversion(self):
        """Test basic depth to point cloud conversion."""
        # Create simple test data
        depth = np.random.rand(100, 100).astype(np.float32)
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        pcd = depth_to_pointcloud(depth, bgr, fx=1000.0, fy=1000.0)

        # Check that point cloud has points and colors
        assert len(pcd.points) > 0
        assert len(pcd.colors) > 0
        assert len(pcd.points) == len(pcd.colors)

    def test_output_dimensions(self):
        """Test that output has correct dimensions."""
        h, w = 50, 80
        depth = np.random.rand(h, w).astype(np.float32)
        bgr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

        pcd = depth_to_pointcloud(depth, bgr, fx=500.0, fy=500.0)

        # Should have at most h*w points (some may be filtered)
        assert len(pcd.points) <= h * w

    def test_color_normalization(self):
        """Test that colors are normalized to [0, 1] range."""
        depth = np.ones((10, 10), dtype=np.float32)
        bgr = np.full((10, 10, 3), 255, dtype=np.uint8)

        pcd = depth_to_pointcloud(depth, bgr, fx=100.0, fy=100.0)

        colors = np.asarray(pcd.colors)
        assert colors.max() <= 1.0
        assert colors.min() >= 0.0

    def test_handles_nan_values(self):
        """Test that NaN values in depth are handled."""
        depth = np.random.rand(20, 20).astype(np.float32)
        depth[5:10, 5:10] = np.nan  # Add NaN region
        bgr = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)

        # Should not raise an error
        pcd = depth_to_pointcloud(depth, bgr, fx=100.0, fy=100.0)

        # Points with NaN should be filtered out
        points = np.asarray(pcd.points)
        assert np.all(np.isfinite(points))


class TestSavePly:
    """Test suite for save_ply function."""

    def test_save_pointcloud(self, tmp_path):
        """Test saving point cloud to PLY file."""
        import open3d as o3d

        # Create a simple point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3))
        pcd.colors = o3d.utility.Vector3dVector(np.random.rand(100, 3))

        output_path = tmp_path / "test.ply"
        save_ply(str(output_path), pcd)

        # Verify file exists and can be loaded
        assert output_path.exists()
        loaded_pcd = o3d.io.read_point_cloud(str(output_path))
        assert len(loaded_pcd.points) == 100
