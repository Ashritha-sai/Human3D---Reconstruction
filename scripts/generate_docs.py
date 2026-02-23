#!/usr/bin/env python3
"""
Generate API documentation for Human3D Gaussian Splatting module.

Usage:
    python scripts/generate_docs.py

This script uses pdoc3 to generate HTML documentation from docstrings.
The output is written to docs/api/.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Generate API documentation."""
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs" / "api"

    # Ensure docs directory exists
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Modules to document
    modules = [
        "src/human3d/reconstruct/gaussian_trainer.py",
        "src/human3d/reconstruct/gaussian_utils.py",
        "src/human3d/reconstruct/losses.py",
        "src/human3d/export/ply_exporter.py",
    ]

    # Generate documentation
    cmd = [
        sys.executable, "-m", "pdoc",
        "--html",
        "--output-dir", str(docs_dir),
        "--force",
    ] + [str(project_root / m) for m in modules]

    print("Generating API documentation...")
    print(f"  Output: {docs_dir}")
    print(f"  Modules: {len(modules)}")
    print()

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        print()
        print("Documentation generated successfully!")
        print()
        print("Generated files:")
        for f in sorted(docs_dir.glob("*.html")):
            print(f"  - {f.name}")

        print()
        print("Open docs/api/index.html to view the documentation.")

    except subprocess.CalledProcessError as e:
        print(f"Error generating documentation: {e}")
        print(e.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
