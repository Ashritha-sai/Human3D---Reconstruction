"""Tests for device selection utilities."""

from unittest.mock import patch

from human3d.utils.device import pick_device


class TestPickDevice:
    """Test suite for pick_device function."""

    def test_prefer_cuda_with_cuda_available(self):
        """Test that CUDA is selected when available and preferred."""
        with patch("torch.cuda.is_available", return_value=True):
            device = pick_device("cuda")
            assert device == "cuda"

    def test_prefer_cuda_without_cuda_available(self):
        """Test fallback to CPU when CUDA is unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            device = pick_device("cuda")
            assert device == "cpu"

    def test_prefer_cpu(self):
        """Test that CPU is selected when explicitly preferred."""
        device = pick_device("cpu")
        assert device == "cpu"

    def test_default_preference(self):
        """Test default preference behavior."""
        with patch("torch.cuda.is_available", return_value=False):
            device = pick_device()
            assert device == "cpu"

    def test_case_insensitive(self):
        """Test that preference is case-insensitive."""
        with patch("torch.cuda.is_available", return_value=True):
            assert pick_device("CUDA") == "cuda"
            assert pick_device("Cuda") == "cuda"

    def test_none_preference(self):
        """Test that None defaults to CUDA preference."""
        with patch("torch.cuda.is_available", return_value=True):
            device = pick_device(None)
            assert device == "cuda"
