"""Tests for configuration loading utilities."""

import pytest

from human3d.utils.config import load_config


class TestLoadConfig:
    """Test suite for load_config function."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid YAML configuration file."""
        config_content = """
        run:
          output_dir: "outputs"
          name_prefix: "run"
        device:
          prefer: "cuda"
        """
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        config = load_config(str(config_file))

        assert config["run"]["output_dir"] == "outputs"
        assert config["run"]["name_prefix"] == "run"
        assert config["device"]["prefer"] == "cuda"

    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_empty_yaml(self, tmp_path):
        """Test loading an empty YAML file returns None."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = load_config(str(config_file))

        assert config is None

    def test_load_nested_config(self, tmp_path):
        """Test loading deeply nested configuration."""
        config_content = """
        segmentation:
          enabled: true
          method: "sam"
          sam:
            model_type: "vit_b"
            checkpoint_path: "checkpoints/sam_vit_b.pth"
            expand_box_px: 30
        """
        config_file = tmp_path / "nested.yaml"
        config_file.write_text(config_content)

        config = load_config(str(config_file))

        assert config["segmentation"]["enabled"] is True
        assert config["segmentation"]["sam"]["model_type"] == "vit_b"
        assert config["segmentation"]["sam"]["expand_box_px"] == 30
