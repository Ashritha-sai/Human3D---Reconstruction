"""Shared test configuration and fixtures for Human3D tests."""

import sys
from pathlib import Path

# Add src to path so all tests can do normal imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
