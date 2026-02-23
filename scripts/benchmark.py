#!/usr/bin/env python3
"""
Benchmark the Gaussian Splatting training pipeline.

Measures:
- Initialization time
- Per-iteration training time
- Rendering time
- Memory usage
- Final quality (PSNR/SSIM)

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --device cuda --iterations 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import torch


def compute_psnr(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    img1 = np.clip(img1.astype(np.float32), 0, 1)
    img2 = np.clip(img2.astype(np.float32), 0, 1)

    if mask is not None:
        mask = mask.astype(bool)
        if mask.ndim == 2:
            mask = mask[:, :, np.newaxis]
        diff = (img1 - img2) ** 2
        mse = diff[np.broadcast_to(mask, img1.shape)].mean()
    else:
        mse = ((img1 - img2) ** 2).mean()

    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index."""
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(img1, img2, channel_axis=2, data_range=1.0))
    except ImportError:
        return 0.0  # skimage not installed


def create_synthetic_data(size: int = 256):
    """Create synthetic test data."""
    np.random.seed(42)

    # Create a simple pattern
    rgb = np.zeros((size, size, 3), dtype=np.uint8)

    # Background gradient
    for y in range(size):
        rgb[y, :, 0] = int(100 + 50 * (y / size))
        rgb[y, :, 1] = int(120 + 30 * (y / size))
        rgb[y, :, 2] = int(150 + 50 * (y / size))

    # Draw colored shapes
    import cv2

    cx, cy = size // 2, size // 2

    # Head (circle)
    cv2.circle(rgb, (cx, cy - 60), 25, (220, 180, 160), -1)

    # Body (rectangle)
    cv2.rectangle(rgb, (cx - 30, cy - 30), (cx + 30, cy + 40), (100, 120, 200), -1)

    # Create depth map (dome shape)
    depth = np.full((size, size), 2.5, dtype=np.float32)
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    person_mask = dist < 80
    depth[person_mask] = 1.5 - 0.3 * np.exp(-dist[person_mask] ** 2 / (2 * 40**2))

    # Create mask
    mask = np.zeros((size, size), dtype=np.uint8)

    # Head
    head_mask = (x - cx) ** 2 + (y - (cy - 60)) ** 2 <= 25**2
    mask[head_mask] = 1

    # Body
    mask[cy - 30 : cy + 40, cx - 30 : cx + 30] = 1

    return rgb, depth, mask


def _import_modules():
    """Import modules directly to avoid ultralytics dependency."""
    import importlib.util
    import types

    project_root = Path(__file__).parent.parent / "src"

    # Import gaussian_utils first
    spec_utils = importlib.util.spec_from_file_location(
        "gaussian_utils",
        project_root / "human3d" / "reconstruct" / "gaussian_utils.py",
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
        project_root / "human3d" / "reconstruct" / "losses.py",
    )
    losses = importlib.util.module_from_spec(spec_losses)
    sys.modules["human3d.reconstruct.losses"] = losses
    sys.modules["human3d.reconstruct"].losses = losses
    spec_losses.loader.exec_module(losses)

    # Import ply_exporter
    spec_ply = importlib.util.spec_from_file_location(
        "human3d.export.ply_exporter",
        project_root / "human3d" / "export" / "ply_exporter.py",
    )
    ply_exporter = importlib.util.module_from_spec(spec_ply)
    sys.modules["human3d.export.ply_exporter"] = ply_exporter
    sys.modules["human3d.export"].ply_exporter = ply_exporter
    spec_ply.loader.exec_module(ply_exporter)

    # Import gaussian_trainer
    spec_trainer = importlib.util.spec_from_file_location(
        "human3d.reconstruct.gaussian_trainer",
        project_root / "human3d" / "reconstruct" / "gaussian_trainer.py",
    )
    gaussian_trainer = importlib.util.module_from_spec(spec_trainer)
    sys.modules["human3d.reconstruct.gaussian_trainer"] = gaussian_trainer
    spec_trainer.loader.exec_module(gaussian_trainer)

    return gaussian_trainer


def benchmark(device: str = "cuda", iterations: int = 500, size: int = 256):
    """Run benchmark."""
    gt = _import_modules()

    GaussianTrainer = gt.GaussianTrainer
    GaussianConfig = gt.GaussianConfig
    CameraParams = gt.CameraParams

    print("=" * 60)
    print("  Gaussian Splatting Benchmark")
    print("=" * 60)

    # Create synthetic data
    print("\n[1/5] Creating test data...")
    rgb, depth, mask = create_synthetic_data(size)
    print(f"  Image size: {size}x{size}")
    print(f"  Masked pixels: {mask.sum()}")

    # Configure
    print("\n[2/5] Configuring...")
    camera = CameraParams(
        fx=500.0,
        fy=500.0,
        cx=(size - 1) / 2.0,
        cy=(size - 1) / 2.0,
        width=size,
        height=size,
    )

    config = GaussianConfig(
        sh_degree=0,
        num_iterations=iterations,
        densify_from_iter=10000,  # Disable densification for benchmark
    )

    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("  [WARN] CUDA not available, using CPU")
        device = "cpu"

    print(f"  Device: {device}")
    print(f"  Iterations: {iterations}")

    # Memory before
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated() / 1024**2
    else:
        mem_before = 0

    # Initialize
    print("\n[3/5] Benchmarking initialization...")
    t0 = time.perf_counter()

    trainer = GaussianTrainer(rgb, depth, mask, camera, config, device=device)
    num_gaussians = trainer.initialize_gaussians()

    t_init = time.perf_counter() - t0
    print(f"  Gaussians: {num_gaussians}")
    print(f"  Init time: {t_init * 1000:.1f} ms")

    # Training
    print("\n[4/5] Benchmarking training...")
    t0 = time.perf_counter()

    history = trainer.optimize(
        num_iterations=iterations,
        log_every=iterations + 1,  # Disable logging
        save_every=0,  # Disable saving
    )

    t_train = time.perf_counter() - t0

    # Memory after
    if device == "cuda":
        mem_after = torch.cuda.max_memory_allocated() / 1024**2
    else:
        import psutil

        mem_after = psutil.Process().memory_info().rss / 1024**2

    # Rendering
    print("\n[5/5] Benchmarking rendering...")
    render_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        trainer.render_view()
        render_times.append(time.perf_counter() - t0)
    t_render = np.mean(render_times)

    # Quality metrics
    rendered_rgb, _, _ = trainer.render_view()
    rendered_np = rendered_rgb.detach().cpu().numpy()
    target_np = rgb.astype(np.float32) / 255.0

    psnr = compute_psnr(rendered_np, target_np, mask)
    ssim = compute_ssim(rendered_np, target_np)

    # Print results
    print("\n" + "=" * 60)
    print("  Benchmark Results")
    print("=" * 60)

    print("\n  Configuration:")
    print(f"    Device: {device}")
    print(f"    Image size: {size}x{size}")
    print(f"    Gaussians: {trainer.num_gaussians}")
    print(f"    Iterations: {iterations}")

    print("\n  Timing:")
    print(f"    Initialization: {t_init * 1000:.1f} ms")
    print(f"    Total training: {t_train:.2f} s")
    print(f"    Per iteration: {t_train / iterations * 1000:.2f} ms")
    print(f"    Rendering: {t_render * 1000:.2f} ms ({1 / t_render:.1f} FPS)")

    print("\n  Memory:")
    print(f"    Peak usage: {mem_after:.1f} MB")

    print("\n  Quality:")
    print(f"    Final loss: {history['loss'][-1]:.4f}")
    print(f"    PSNR: {psnr:.2f} dB")
    print(f"    SSIM: {ssim:.4f}")

    print("\n" + "=" * 60)

    return {
        "device": device,
        "size": size,
        "iterations": iterations,
        "num_gaussians": trainer.num_gaussians,
        "init_time_ms": t_init * 1000,
        "train_time_s": t_train,
        "iter_time_ms": t_train / iterations * 1000,
        "render_time_ms": t_render * 1000,
        "render_fps": 1 / t_render,
        "memory_mb": mem_after,
        "final_loss": history["loss"][-1],
        "psnr": psnr,
        "ssim": ssim,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gaussian Splatting")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to benchmark (default: cuda)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of iterations (default: 500)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Image size (default: 256)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare CPU vs GPU performance",
    )

    args = parser.parse_args()

    if args.compare:
        print("\nBenchmarking CPU...")
        cpu_results = benchmark("cpu", args.iterations, args.size)

        if torch.cuda.is_available():
            print("\nBenchmarking GPU...")
            gpu_results = benchmark("cuda", args.iterations, args.size)

            print("\n" + "=" * 60)
            print("  CPU vs GPU Comparison")
            print("=" * 60)
            print(f"\n  Speedup (training): {cpu_results['train_time_s'] / gpu_results['train_time_s']:.1f}x")
            print(f"  Speedup (rendering): {cpu_results['render_time_ms'] / gpu_results['render_time_ms']:.1f}x")
        else:
            print("\n[INFO] CUDA not available, GPU benchmark skipped")
    else:
        benchmark(args.device, args.iterations, args.size)


if __name__ == "__main__":
    main()
