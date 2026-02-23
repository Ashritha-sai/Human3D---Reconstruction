"""
Generate test fixture data for end-to-end tests.

Run this script to generate synthetic test data:
    python tests/fixtures/generate_fixtures.py
"""

import numpy as np
import cv2
import yaml
from pathlib import Path


def generate_synthetic_person_image(output_dir: Path):
    """Generate a synthetic person-like image for testing."""
    H, W = 256, 256

    # Create RGB image with a person silhouette
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Background gradient
    for y in range(H):
        rgb[y, :, 0] = int(100 + 50 * (y / H))  # Red gradient
        rgb[y, :, 1] = int(120 + 30 * (y / H))  # Green gradient
        rgb[y, :, 2] = int(150 + 50 * (y / H))  # Blue gradient

    # Draw a simple person shape (head, body, arms, legs)
    cx, cy = W // 2, H // 2

    # Head (circle)
    cv2.circle(rgb, (cx, cy - 60), 25, (220, 180, 160), -1)

    # Body (rectangle)
    cv2.rectangle(rgb, (cx - 30, cy - 30), (cx + 30, cy + 40), (100, 120, 200), -1)

    # Arms (rectangles)
    cv2.rectangle(rgb, (cx - 55, cy - 25), (cx - 30, cy + 10), (220, 180, 160), -1)
    cv2.rectangle(rgb, (cx + 30, cy - 25), (cx + 55, cy + 10), (220, 180, 160), -1)

    # Legs (rectangles)
    cv2.rectangle(rgb, (cx - 25, cy + 40), (cx - 5, cy + 90), (50, 50, 150), -1)
    cv2.rectangle(rgb, (cx + 5, cy + 40), (cx + 25, cy + 90), (50, 50, 150), -1)

    # Save RGB image
    cv2.imwrite(str(output_dir / "person.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return rgb


def generate_synthetic_depth(output_dir: Path, H=256, W=256):
    """Generate synthetic depth map for testing."""
    depth = np.full((H, W), 2.5, dtype=np.float32)

    # Create dome shape for person
    cx, cy = W // 2, H // 2
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Person is closer (lower depth values)
    person_mask = dist < 80
    depth[person_mask] = 1.5 - 0.3 * np.exp(-dist[person_mask] ** 2 / (2 * 40 ** 2))

    # Save depth
    np.save(str(output_dir / "person_depth.npy"), depth)

    return depth


def generate_synthetic_mask(output_dir: Path, H=256, W=256):
    """Generate person mask for testing."""
    mask = np.zeros((H, W), dtype=np.uint8)

    cx, cy = W // 2, H // 2

    # Head
    y, x = np.ogrid[:H, :W]
    head_mask = (x - cx) ** 2 + (y - (cy - 60)) ** 2 <= 25 ** 2
    mask[head_mask] = 1

    # Body
    mask[cy - 30:cy + 40, cx - 30:cx + 30] = 1

    # Arms
    mask[cy - 25:cy + 10, cx - 55:cx - 30] = 1
    mask[cy - 25:cy + 10, cx + 30:cx + 55] = 1

    # Legs
    mask[cy + 40:cy + 90, cx - 25:cx - 5] = 1
    mask[cy + 40:cy + 90, cx + 5:cx + 25] = 1

    # Save mask
    np.save(str(output_dir / "person_mask.npy"), mask)

    # Also save as image for visualization
    cv2.imwrite(str(output_dir / "person_mask.png"), mask * 255)

    return mask


def generate_camera_params(output_dir: Path, H=256, W=256):
    """Generate camera parameters for testing."""
    params = {
        'fx': 500.0,
        'fy': 500.0,
        'cx': (W - 1) / 2.0,
        'cy': (H - 1) / 2.0,
        'width': W,
        'height': H,
    }

    with open(output_dir / "camera_params.yaml", 'w') as f:
        yaml.dump(params, f, default_flow_style=False)

    return params


def main():
    output_dir = Path(__file__).parent
    print(f"Generating fixtures in: {output_dir}")

    print("1. Generating synthetic person image...")
    rgb = generate_synthetic_person_image(output_dir)
    print(f"   Saved: person.jpg ({rgb.shape})")

    print("2. Generating synthetic depth map...")
    depth = generate_synthetic_depth(output_dir)
    print(f"   Saved: person_depth.npy ({depth.shape})")

    print("3. Generating person mask...")
    mask = generate_synthetic_mask(output_dir)
    print(f"   Saved: person_mask.npy ({mask.shape})")
    print(f"   Masked pixels: {mask.sum()}")

    print("4. Generating camera parameters...")
    params = generate_camera_params(output_dir)
    print(f"   Saved: camera_params.yaml")

    print("\nFixtures generated successfully!")


if __name__ == "__main__":
    main()
