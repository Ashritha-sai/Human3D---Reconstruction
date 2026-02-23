"""
Gaussian Splatting Trainer Module

This module provides the core training loop for 3D Gaussian Splatting reconstruction
from a single RGB-D image with segmentation mask. It handles initialization of
Gaussian primitives from depth data and optimizes their parameters to minimize
photometric loss.

The implementation uses gsplat for efficient differentiable rasterization.

References:
    - 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
    - gsplat library: https://github.com/nerfstudio-project/gsplat
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class CameraParams:
    """
    Camera intrinsic and extrinsic parameters.

    Attributes:
        fx: Focal length in x direction (pixels).
        fy: Focal length in y direction (pixels).
        cx: Principal point x coordinate (pixels).
        cy: Principal point y coordinate (pixels).
        width: Image width in pixels.
        height: Image height in pixels.
        world_to_camera: 4x4 transformation matrix from world to camera coordinates.
            Default is identity (camera at origin looking down -Z).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    world_to_camera: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.world_to_camera is None:
            self.world_to_camera = np.eye(4, dtype=np.float32)


@dataclass
class GaussianConfig:
    """
    Configuration for Gaussian Splatting training.

    Attributes:
        num_iterations: Total number of optimization iterations.
        lr_position: Learning rate for Gaussian center positions (xyz).
        lr_color: Learning rate for spherical harmonic coefficients.
        lr_scale: Learning rate for Gaussian scales (log-space).
        lr_opacity: Learning rate for opacity (logit-space).
        lr_rotation: Learning rate for rotation quaternions.
        loss_weight_l1: Weight for L1 photometric loss.
        loss_weight_ssim: Weight for SSIM structural loss.
        loss_weight_lpips: Weight for LPIPS perceptual loss.
        densify_from_iter: Start densification after this iteration.
        densify_until_iter: Stop densification after this iteration.
        densify_interval: Iterations between densification steps.
        densify_grad_threshold: Gradient threshold for densification.
        prune_opacity_threshold: Minimum opacity before pruning.
        max_gaussians: Maximum number of Gaussians allowed.
        scale_init: Initial scale for Gaussians (meters).
        opacity_init: Initial opacity value [0, 1].
        position_noise: Random noise added to initial positions (meters).
        sh_degree: Spherical harmonics degree (0 = DC only, 3 = full).
    """

    num_iterations: int = 1000
    lr_position: float = 0.001
    lr_color: float = 0.01
    lr_scale: float = 0.005
    lr_opacity: float = 0.01
    lr_rotation: float = 0.001
    loss_weight_l1: float = 0.8
    loss_weight_ssim: float = 0.2
    loss_weight_lpips: float = 0.0
    # Densification parameters
    densify_from_iter: int = 500
    densify_until_iter: int = 3000
    densify_interval: int = 100
    densify_grad_threshold: float = 0.0002
    prune_opacity_threshold: float = 0.005
    max_gaussians: int = 100000
    # Initialization parameters
    scale_init: float = 0.01
    opacity_init: float = 0.5
    position_noise: float = 0.001
    sh_degree: int = 0


class GaussianTrainer:
    """
    Trainer for 3D Gaussian Splatting from a single RGB-D image.

    This class handles:
    1. Initialization of Gaussian primitives from depth map
    2. Optimization of Gaussian parameters (position, color, scale, opacity, rotation)
    3. Adaptive density control (densification and pruning)
    4. Rendering novel views
    5. Exporting trained Gaussians to PLY format

    The Gaussian representation consists of:
        - means: (N, 3) center positions in world coordinates
        - scales: (N, 3) axis-aligned scales (log-space during optimization)
        - rotations: (N, 4) rotation quaternions (wxyz format)
        - sh_coeffs: (N, C, 3) spherical harmonic coefficients for view-dependent color
        - opacities: (N,) opacity values (logit-space during optimization)

    Attributes:
        rgb: Original RGB image as torch.Tensor, shape (H, W, 3), range [0, 1].
        depth: Depth map as torch.Tensor, shape (H, W), in meters.
        mask: Binary segmentation mask as torch.Tensor, shape (H, W), {0, 1}.
        camera: CameraParams instance with intrinsics and pose.
        config: GaussianConfig instance with training hyperparameters.
        device: torch.device for computation ('cuda' or 'cpu').

    Example:
        >>> trainer = GaussianTrainer(rgb, depth, mask, camera_params, config)
        >>> trainer.initialize_gaussians()
        >>> trainer.optimize(num_iterations=1000)
        >>> trainer.export_ply("output/gaussians.ply")
    """

    def __init__(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        camera_params: CameraParams,
        config: GaussianConfig,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the Gaussian trainer with input data.

        Args:
            rgb: Input RGB image.
                Shape: (H, W, 3)
                dtype: uint8 or float32
                Range: [0, 255] for uint8, [0, 1] for float32

            depth: Depth map (metric or relative).
                Shape: (H, W)
                dtype: float32
                Range: Positive values, higher = farther

            mask: Binary segmentation mask indicating subject pixels.
                Shape: (H, W)
                dtype: uint8 or bool
                Values: 0 = background, 1 = subject

            camera_params: Camera intrinsic and extrinsic parameters.
                See CameraParams dataclass for details.

            config: Training configuration.
                See GaussianConfig dataclass for details.

            device: Computation device, either 'cuda' or 'cpu'.
                Default: 'cuda'

        Raises:
            ValueError: If input shapes are incompatible.
            RuntimeError: If CUDA is requested but not available.
        """
        # Validate device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, falling back to CPU")
            device = "cpu"
        self.device = device

        # Validate input shapes
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB must have shape (H, W, 3), got {rgb.shape}")
        if depth.ndim != 2:
            raise ValueError(f"Depth must have shape (H, W), got {depth.shape}")
        if mask.ndim != 2:
            raise ValueError(f"Mask must have shape (H, W), got {mask.shape}")

        h, w = depth.shape
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"RGB shape {rgb.shape[:2]} doesn't match depth shape {(h, w)}")
        if mask.shape != (h, w):
            raise ValueError(f"Mask shape {mask.shape} doesn't match depth shape {(h, w)}")

        # Convert RGB to float32 [0, 1] if needed
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        elif rgb.max() > 1.0:
            rgb = rgb.astype(np.float32) / 255.0

        # Convert to torch tensors
        self.rgb = torch.from_numpy(rgb.astype(np.float32)).to(device)  # (H, W, 3)
        self.depth = torch.from_numpy(depth.astype(np.float32)).to(device)  # (H, W)
        self.mask = torch.from_numpy(mask.astype(np.float32)).to(device)  # (H, W)

        # Store camera params and config
        self.camera = camera_params
        self.config = config

        # Initialize placeholders for Gaussian parameters (set by initialize_gaussians)
        self.means: Optional[nn.Parameter] = None
        self.scales: Optional[nn.Parameter] = None
        self.rotations: Optional[nn.Parameter] = None
        self.sh_coeffs: Optional[nn.Parameter] = None
        self.opacities: Optional[nn.Parameter] = None

        # Optimizer (set by initialize_gaussians)
        self.optimizer: Optional[torch.optim.Adam] = None

        # For tracking gradients during densification
        self._xyz_gradient_accum: Optional[Tensor] = None
        self._denom: Optional[Tensor] = None

    def initialize_gaussians(self) -> int:
        """
        Initialize Gaussian primitives from depth map and mask.

        This method:
        1. Unprojects masked depth pixels to 3D points using camera intrinsics
        2. Samples RGB colors from corresponding pixels
        3. Estimates initial scales based on local point density
        4. Sets default rotation (identity quaternion) and opacity

        The number of Gaussians equals the number of valid masked pixels.

        Returns:
            int: Number of Gaussians initialized (N).

        Side Effects:
            Initializes the following trainable parameters:
                - self.means: (N, 3) Gaussian centers
                - self.scales: (N, 3) log-scale parameters
                - self.rotations: (N, 4) quaternions (wxyz)
                - self.sh_coeffs: (N, 1, 3) or (N, 16, 3) SH coefficients
                - self.opacities: (N,) logit-opacity values

        Mathematical Operations:
            Unprojection from pixel (u, v) with depth d:
                X = (u - cx) * d / fx
                Y = (v - cy) * d / fy
                Z = d

            Scale estimation via k-NN:
                scale_i = mean(||p_i - p_neighbors||) / 2

            Opacity initialization:
                logit_opacity = log(opacity_init / (1 - opacity_init))

        Example:
            >>> num_gaussians = trainer.initialize_gaussians()
            >>> print(f"Initialized {num_gaussians} Gaussians")
        """
        from .gaussian_utils import (
            normalize_depth_to_metric,
            depth_to_xyz,
            estimate_point_scales,
            rgb_to_spherical_harmonics,
            inverse_sigmoid,
        )

        # Normalize depth to pseudo-metric range (same as original pointcloud.py)
        depth_normalized = normalize_depth_to_metric(self.depth, min_depth=0.5, max_depth=2.5)

        # Get 3D points from depth (using mask to filter)
        xyz = depth_to_xyz(
            depth_normalized,
            self.mask,
            fx=self.camera.fx,
            fy=self.camera.fy,
            cx=self.camera.cx,
            cy=self.camera.cy,
        )

        n_points = xyz.shape[0]
        if n_points == 0:
            raise ValueError("No valid points found in masked region")

        # Add small position noise for better optimization
        if self.config.position_noise > 0:
            noise = torch.randn_like(xyz) * self.config.position_noise
            xyz = xyz + noise

        # Sample RGB colors from image at masked pixel locations
        # Get the mask indices
        mask_bool = self.mask > 0
        valid_mask = mask_bool & (depth_normalized > 0) & torch.isfinite(depth_normalized)

        # Flatten and get valid indices
        rgb_flat = self.rgb.reshape(-1, 3)  # (H*W, 3)
        valid_flat = valid_mask.reshape(-1)

        rgb_colors = rgb_flat[valid_flat]  # (N, 3)

        # Convert RGB to spherical harmonics
        sh_degree = self.config.sh_degree
        sh_coeffs = rgb_to_spherical_harmonics(rgb_colors, degree=sh_degree)  # (N, num_sh, 3)

        # Estimate scales using k-NN
        scales = estimate_point_scales(xyz, k_neighbors=8)  # (N, 3)

        # Convert scales to log-space for optimization
        log_scales = torch.log(scales + 1e-8)

        # Initialize rotations as identity quaternions (w, x, y, z) = (1, 0, 0, 0)
        rotations = torch.zeros((n_points, 4), dtype=torch.float32, device=self.device)
        rotations[:, 0] = 1.0  # w = 1

        # Initialize opacities in logit space
        # Default opacity = 0.9
        opacity_init = self.config.opacity_init
        # Clamp to avoid infinity
        opacity_init = max(0.01, min(0.99, opacity_init))
        logit_opacity = inverse_sigmoid(
            torch.full((n_points,), opacity_init, dtype=torch.float32, device=self.device)
        )

        # Create trainable parameters
        self.means = nn.Parameter(xyz.contiguous())
        self.scales = nn.Parameter(log_scales.contiguous())
        self.rotations = nn.Parameter(rotations.contiguous())
        self.sh_coeffs = nn.Parameter(sh_coeffs.contiguous())
        self.opacities = nn.Parameter(logit_opacity.contiguous())

        # Set up optimizer with per-parameter learning rates
        self.optimizer = torch.optim.Adam(
            [
                {"params": [self.means], "lr": self.config.lr_position, "name": "means"},
                {"params": [self.scales], "lr": self.config.lr_scale, "name": "scales"},
                {"params": [self.rotations], "lr": self.config.lr_rotation, "name": "rotations"},
                {"params": [self.sh_coeffs], "lr": self.config.lr_color, "name": "sh_coeffs"},
                {"params": [self.opacities], "lr": self.config.lr_opacity, "name": "opacities"},
            ]
        )

        # Initialize gradient accumulators for densification
        self._xyz_gradient_accum = torch.zeros((n_points, 1), device=self.device)
        self._denom = torch.zeros((n_points, 1), device=self.device)

        return n_points

    def optimize(
        self,
        num_iterations: Optional[int] = None,
        log_every: int = 50,
        save_every: int = 500,
        output_dir: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Run the main optimization loop.

        Optimizes Gaussian parameters to minimize photometric loss between
        rendered and target images. Periodically performs densification
        and pruning to improve reconstruction quality.

        Args:
            num_iterations: Number of optimization steps.
                If None, uses config.num_iterations.
                Default: None

            log_every: Print loss every N iterations.
                Default: 50

            save_every: Save rendered image every N iterations.
                Default: 500

            output_dir: Directory to save rendered images.
                If None, uses './outputs'.
                Default: None

        Returns:
            Dict[str, list]: Training history containing:
                - 'loss': Total loss per iteration
                - 'l1': L1 component per iteration
                - 'ssim': SSIM component per iteration
                - 'num_gaussians': Gaussian count per iteration
                - 'iteration': Iteration numbers

        Side Effects:
            - Updates all trainable Gaussian parameters
            - May change number of Gaussians via densification/pruning
            - Prints progress every log_every iterations
            - Saves rendered images every save_every iterations

        Training Loop:
            For each iteration:
                1. Render image from training viewpoint
                2. Compute loss (L1 + SSIM + optional LPIPS)
                3. Backpropagate gradients
                4. Update parameters via optimizer
                5. If iteration % densify_interval == 0:
                   - Densify Gaussians with high gradient
                   - Prune Gaussians with low opacity

        Example:
            >>> history = trainer.optimize(num_iterations=1000)
            >>> plt.plot(history['loss'])
            >>> plt.title('Training Loss')
        """
        from .losses import GaussianLosses

        # Validate state
        if self.means is None:
            raise RuntimeError("Must call initialize_gaussians() before optimize()")

        # Set up parameters
        if num_iterations is None:
            num_iterations = self.config.num_iterations

        if output_dir is None:
            output_dir = Path("./outputs")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create optimizer with recommended learning rates from 3DGS paper.
        # These learning rates are empirically tuned:
        # - means (1.6e-4): Slow position updates prevent oscillation
        # - scales (5e-3): Moderate rate allows size adjustment
        # - rotations (1e-3): Careful orientation updates for stability
        # - sh_coeffs (2.5e-3): Balance color refinement speed
        # - opacities (5e-2): Fast transparency updates for pruning
        self.optimizer = torch.optim.Adam(
            [
                {"params": [self.means], "lr": 1.6e-4, "name": "means"},
                {"params": [self.scales], "lr": 5e-3, "name": "scales"},
                {"params": [self.rotations], "lr": 1e-3, "name": "rotations"},
                {"params": [self.sh_coeffs], "lr": 2.5e-3, "name": "sh_coeffs"},
                {"params": [self.opacities], "lr": 5e-2, "name": "opacities"},
            ]
        )

        # Create loss function
        losses = GaussianLosses(
            weight_l1=self.config.loss_weight_l1,
            weight_ssim=self.config.loss_weight_ssim,
            weight_lpips=self.config.loss_weight_lpips,
            weight_scale_reg=0.01,
            weight_opacity_reg=0.01,
            device=self.device,
        )

        # Initialize history
        history = {
            "loss": [],
            "l1": [],
            "ssim": [],
            "lpips": [],
            "num_gaussians": [],
            "iteration": [],
            "densify_added": [],
            "densify_removed": [],
        }

        # Target image and mask
        target_rgb = self.rgb  # (H, W, 3)
        mask = self.mask  # (H, W)

        # Initialize gradient accumulators
        self._xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device=self.device)
        self._denom = torch.zeros((self.num_gaussians, 1), device=self.device)

        # Densification settings
        densify_from = self.config.densify_from_iter
        densify_until = self.config.densify_until_iter
        densify_interval = self.config.densify_interval

        # Try to import tqdm for progress bars
        try:
            from tqdm import tqdm

            use_tqdm = True
        except ImportError:
            use_tqdm = False

        # Mixed precision support for GPU training
        use_amp = self.device != "cpu" and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        print(f"Starting optimization for {num_iterations} iterations")
        print(f"  Gaussians: {self.num_gaussians}")
        print(f"  Image size: {target_rgb.shape[0]}x{target_rgb.shape[1]}")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {'enabled' if use_amp else 'disabled'}")
        print(f"  Densification: iter {densify_from} to {densify_until}, every {densify_interval}")
        print("-" * 50)

        # Training loop with optional progress bar
        iterator = range(num_iterations)
        if use_tqdm:
            iterator = tqdm(iterator, desc="Training", unit="iter", ncols=100)

        for iteration in iterator:
            self.optimizer.zero_grad()

            # Forward pass with optional mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    rendered_rgb, rendered_depth, rendered_alpha = self.render_view()
                    total_loss, components = losses.total_loss(
                        rendered=rendered_rgb,
                        target=target_rgb,
                        mask=mask,
                        scales=self.scales,
                        opacities=self.opacities,
                    )
            else:
                rendered_rgb, rendered_depth, rendered_alpha = self.render_view()
                total_loss, components = losses.total_loss(
                    rendered=rendered_rgb,
                    target=target_rgb,
                    mask=mask,
                    scales=self.scales,
                    opacities=self.opacities,
                )

            # Check for NaN
            if torch.isnan(total_loss):
                if use_tqdm:
                    iterator.close()
                print(f"\n[WARN] NaN loss at iteration {iteration}, stopping")
                break

            # Backward pass with optional mixed precision
            if use_amp and scaler is not None:
                scaler.scale(total_loss).backward()
                self._accumulate_gradients()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [self.means, self.scales, self.rotations, self.sh_coeffs, self.opacities],
                    max_norm=1.0,
                )
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                self._accumulate_gradients()
                torch.nn.utils.clip_grad_norm_(
                    [self.means, self.scales, self.rotations, self.sh_coeffs, self.opacities],
                    max_norm=1.0,
                )
                self.optimizer.step()

            # Normalize quaternions after update
            with torch.no_grad():
                self.rotations.data = self.rotations.data / (
                    torch.norm(self.rotations.data, dim=-1, keepdim=True) + 1e-8
                )

            # Densification and pruning
            num_added, num_removed = 0, 0
            if densify_from <= iteration < densify_until and iteration % densify_interval == 0:
                if iteration > 0:
                    num_added, num_removed = self._densify_and_prune()

            # Record history
            history["loss"].append(components.get("total", total_loss.item()))
            history["l1"].append(components.get("l1", 0.0))
            history["ssim"].append(components.get("ssim", 0.0))
            history["lpips"].append(components.get("lpips", 0.0))
            history["num_gaussians"].append(self.num_gaussians)
            history["iteration"].append(iteration)
            history["densify_added"].append(num_added)
            history["densify_removed"].append(num_removed)

            # Update progress bar
            if use_tqdm:
                iterator.set_postfix(
                    loss=f"{total_loss.item():.4f}",
                    n=self.num_gaussians,
                    refresh=False,
                )
            elif iteration % log_every == 0 or iteration == num_iterations - 1:
                loss_str = f"Iter {iteration:5d}/{num_iterations}"
                loss_str += f" | Loss: {total_loss.item():.4f}"
                loss_str += f" | L1: {components.get('l1', 0):.4f}"
                loss_str += f" | SSIM: {components.get('ssim', 0):.4f}"
                loss_str += f" | Gaussians: {self.num_gaussians}"
                print(loss_str)

            # Save rendered image
            if save_every > 0 and (iteration % save_every == 0 or iteration == num_iterations - 1):
                self._save_iteration_image(rendered_rgb, target_rgb, iteration, output_dir)

        print("-" * 50)
        print("Optimization complete!")
        print(f"  Final loss: {history['loss'][-1]:.4f}")
        print(f"  Gaussians: {self.num_gaussians}")

        return history

    def _save_iteration_image(
        self,
        rendered: Tensor,
        target: Tensor,
        iteration: int,
        output_dir: Path,
    ) -> None:
        """Save rendered image for visualization."""
        import cv2

        # Detach and convert to numpy
        rendered_np = rendered.detach().cpu().numpy()
        rendered_np = np.clip(rendered_np, 0, 1)

        # Convert RGB to BGR for OpenCV
        rendered_bgr = (rendered_np[:, :, ::-1] * 255).astype(np.uint8)

        # Save
        filename = output_dir / f"iteration_{iteration:05d}.png"
        cv2.imwrite(str(filename), rendered_bgr)

        # Also save comparison image
        target_np = target.detach().cpu().numpy()
        target_np = np.clip(target_np, 0, 1)

        # Side by side: target | rendered | difference
        diff_np = np.abs(target_np - rendered_np)
        diff_enhanced = np.clip(diff_np * 5.0, 0, 1)

        comparison = np.concatenate([target_np, rendered_np, diff_enhanced], axis=1)
        comparison_bgr = (comparison[:, :, ::-1] * 255).astype(np.uint8)

        comparison_filename = output_dir / f"comparison_{iteration:05d}.png"
        cv2.imwrite(str(comparison_filename), comparison_bgr)

    def render_view(
        self,
        camera_pose: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraParams] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Render the Gaussian splats from a given camera viewpoint.

        Uses differentiable rasterization via gsplat to project 3D Gaussians
        to a 2D image. Supports rendering from the training view or novel views.

        Args:
            camera_pose: 4x4 world-to-camera transformation matrix.
                Shape: (4, 4)
                dtype: float32
                If None, uses the training camera pose.
                Default: None

            intrinsics: Camera intrinsic parameters for rendering.
                If None, uses the training camera intrinsics.
                Default: None

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Rendered outputs
                - rgb: Rendered RGB image
                    Shape: (H, W, 3)
                    Range: [0, 1]

                - depth: Rendered depth map
                    Shape: (H, W)
                    Range: [0, inf) meters

                - alpha: Rendered alpha/opacity map
                    Shape: (H, W)
                    Range: [0, 1]

        Mathematical Operations:
            For each Gaussian i:
                1. Transform mean to camera space: p_cam = R @ p_world + t
                2. Project to image plane: u = fx * x/z + cx, v = fy * y/z + cy
                3. Compute 2D covariance from 3D covariance + Jacobian
                4. Evaluate Gaussian contribution per pixel
                5. Alpha-composite front-to-back

        Example:
            >>> rgb, depth, alpha = trainer.render_view()
            >>> plt.imshow(rgb.cpu().numpy())

            >>> # Render from rotated viewpoint
            >>> R = rotation_matrix_y(np.pi / 6)  # 30 degree rotation
            >>> pose = np.eye(4)
            >>> pose[:3, :3] = R
            >>> rgb_novel, _, _ = trainer.render_view(camera_pose=pose)
        """
        from .gaussian_utils import (
            create_camera_intrinsics,
            create_front_camera_pose,
            sigmoid,
        )

        # Get camera parameters
        if intrinsics is None:
            intrinsics = self.camera

        H, W = intrinsics.height, intrinsics.width
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.cx, intrinsics.cy

        # Get view matrix
        if camera_pose is None:
            viewmat = create_front_camera_pose(device=self.device)
        else:
            if isinstance(camera_pose, np.ndarray):
                viewmat = torch.tensor(camera_pose, dtype=torch.float32, device=self.device)
            else:
                viewmat = camera_pose.to(self.device)

        # Get intrinsics matrix
        K = create_camera_intrinsics(fx, fy, cx, cy, device=self.device)

        # Get Gaussian parameters
        means = self.means  # (N, 3)
        quats = self.rotations  # (N, 4) wxyz
        scales = torch.exp(self.scales)  # (N, 3) - convert from log-space
        opacities = sigmoid(self.opacities)  # (N,) - convert from logit-space
        sh_coeffs = self.sh_coeffs  # (N, num_sh, 3)

        # Normalize quaternions
        quats = quats / (torch.norm(quats, dim=-1, keepdim=True) + 1e-8)

        # Check if we can use gsplat (requires CUDA)
        use_gsplat = self.device != "cpu" and torch.cuda.is_available()

        if use_gsplat:
            try:
                return self._render_gsplat(
                    means, quats, scales, opacities, sh_coeffs, viewmat, K, H, W
                )
            except Exception as e:
                print(f"[WARN] gsplat rendering failed: {e}, falling back to CPU renderer")
                use_gsplat = False

        # CPU fallback renderer
        return self._render_cpu_fallback(
            means, quats, scales, opacities, sh_coeffs, viewmat, K, H, W
        )

    def _render_gsplat(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        sh_coeffs: Tensor,
        viewmat: Tensor,
        K: Tensor,
        H: int,
        W: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Render using gsplat library (requires CUDA)."""
        import os

        # Set CUDA_HOME if not already set, to help gsplat find CUDA toolkit
        if "CUDA_HOME" not in os.environ and "CUDA_PATH" not in os.environ:
            cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
            ]
            for cuda_path in cuda_paths:
                if os.path.exists(cuda_path):
                    os.environ["CUDA_HOME"] = cuda_path
                    break

        from gsplat import rasterization

        # Prepare inputs for gsplat
        # gsplat expects: viewmats (C, 4, 4), Ks (C, 3, 3)
        viewmats = viewmat.unsqueeze(0)  # (1, 4, 4)
        Ks = K.unsqueeze(0)  # (1, 3, 3)

        # Determine SH degree from coefficient shape
        num_sh = sh_coeffs.shape[1]
        sh_degree = int(np.sqrt(num_sh)) - 1

        # Rasterize
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_coeffs,  # (N, K, 3) SH coefficients
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            near_plane=0.01,
            far_plane=100.0,
            render_mode="RGB+D",
            sh_degree=sh_degree if sh_degree >= 0 else None,
        )

        # Extract outputs
        # render_colors shape: (1, H, W, 4) for RGB+D mode
        rgb = render_colors[0, :, :, :3]  # (H, W, 3)
        depth = render_colors[0, :, :, 3]  # (H, W)
        alpha = render_alphas[0, :, :, 0]  # (H, W)

        # Clamp RGB to [0, 1]
        rgb = torch.clamp(rgb, 0.0, 1.0)

        return rgb, depth, alpha

    def _render_cpu_fallback(
        self,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        opacities: Tensor,
        sh_coeffs: Tensor,
        viewmat: Tensor,
        K: Tensor,
        H: int,
        W: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Simple CPU-based splatting renderer for testing.

        This is a simplified renderer that projects Gaussians as circles.
        It's much slower and less accurate than gsplat but works on CPU.
        """
        from .gaussian_utils import spherical_harmonics_to_rgb

        device = means.device

        # Initialize output buffers
        rgb_buffer = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
        depth_buffer = torch.full((H, W), float("inf"), dtype=torch.float32, device=device)
        alpha_buffer = torch.zeros((H, W), dtype=torch.float32, device=device)

        # Extract camera parameters
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Transform means to camera space
        R = viewmat[:3, :3]  # (3, 3)
        t = viewmat[:3, 3]  # (3,)
        means_cam = means @ R.T + t  # (N, 3)

        # Filter points behind camera
        valid = means_cam[:, 2] > 0.1
        if not valid.any():
            return rgb_buffer, depth_buffer, alpha_buffer

        means_cam = means_cam[valid]
        scales_valid = scales[valid]
        opacities_valid = opacities[valid]
        sh_coeffs_valid = sh_coeffs[valid]

        # Project to image plane
        z = means_cam[:, 2]
        u = fx * means_cam[:, 0] / z + cx  # (N,)
        v = fy * means_cam[:, 1] / z + cy  # (N,)

        # Get colors from SH (DC term only for CPU fallback)
        colors = spherical_harmonics_to_rgb(sh_coeffs_valid)  # (N, 3)

        # Compute approximate 2D radius from 3D scale
        # Use average scale and project
        avg_scale = scales_valid.mean(dim=1)  # (N,)
        radius_2d = (fx * avg_scale / z).clamp(min=1, max=50)  # (N,)

        # Sort by depth (front to back for proper compositing)
        depth_order = torch.argsort(z)

        # Splat each Gaussian
        for idx in depth_order:
            px, py = int(u[idx].item()), int(v[idx].item())
            r = int(radius_2d[idx].item())
            color = colors[idx]
            opacity = opacities_valid[idx].item()
            d = z[idx].item()

            # Bounds check
            x_min = max(0, px - r)
            x_max = min(W, px + r + 1)
            y_min = max(0, py - r)
            y_max = min(H, py + r + 1)

            if x_min >= x_max or y_min >= y_max:
                continue

            # Create coordinate grids for the patch
            yy, xx = torch.meshgrid(
                torch.arange(y_min, y_max, device=device),
                torch.arange(x_min, x_max, device=device),
                indexing="ij",
            )

            # Compute Gaussian weights
            dist_sq = (xx - px).float() ** 2 + (yy - py).float() ** 2
            sigma = r / 2.0
            weights = torch.exp(-dist_sq / (2 * sigma**2)) * opacity

            # Alpha compositing
            alpha_contrib = weights * (1 - alpha_buffer[y_min:y_max, x_min:x_max])

            rgb_buffer[y_min:y_max, x_min:x_max] += alpha_contrib.unsqueeze(-1) * color
            alpha_buffer[y_min:y_max, x_min:x_max] += alpha_contrib

            # Update depth (keep minimum)
            depth_mask = weights > 0.01
            depth_buffer[y_min:y_max, x_min:x_max] = torch.where(
                depth_mask & (d < depth_buffer[y_min:y_max, x_min:x_max]),
                torch.full_like(depth_buffer[y_min:y_max, x_min:x_max], d),
                depth_buffer[y_min:y_max, x_min:x_max],
            )

        # Normalize by alpha
        rgb_buffer = torch.clamp(rgb_buffer, 0.0, 1.0)

        return rgb_buffer, depth_buffer, alpha_buffer

    def export_ply(self, output_path: str | Path) -> None:
        """
        Export trained Gaussians to PLY format.

        Saves Gaussian parameters in a format compatible with standard
        3D Gaussian Splatting viewers. The PLY file contains:
            - positions (xyz)
            - colors (RGB or SH coefficients)
            - scales (3D)
            - rotations (quaternion)
            - opacities

        Args:
            output_path: Path to save the PLY file.
                Will create parent directories if needed.
                Should have .ply extension.

        Side Effects:
            Creates a PLY file at the specified path.

        File Format:
            The PLY file contains the following properties per vertex:
                - x, y, z: Position coordinates
                - f_dc_0, f_dc_1, f_dc_2: DC spherical harmonic (color)
                - f_rest_0 ... f_rest_44: Higher-order SH (if sh_degree > 0)
                - opacity: Opacity value (sigmoid-activated)
                - scale_0, scale_1, scale_2: Log-scale values
                - rot_0, rot_1, rot_2, rot_3: Rotation quaternion (wxyz)

        Example:
            >>> trainer.export_ply("outputs/trained_gaussians.ply")
        """
        from ..export.ply_exporter import save_gaussian_ply
        from .gaussian_utils import sigmoid

        if self.means is None:
            raise RuntimeError("No Gaussians to export. Call initialize_gaussians() first.")

        # Convert to numpy
        means_np = self.means.detach().cpu().numpy()
        scales_np = self.scales.detach().cpu().numpy()  # Keep in log-space
        rotations_np = self.rotations.detach().cpu().numpy()
        sh_coeffs_np = self.sh_coeffs.detach().cpu().numpy()

        # Apply sigmoid to opacities
        opacities_np = sigmoid(self.opacities).detach().cpu().numpy()

        # Save PLY
        save_gaussian_ply(
            means=means_np,
            scales=scales_np,
            rotations=rotations_np,
            sh_coeffs=sh_coeffs_np,
            opacities=opacities_np,
            filepath=output_path,
        )

        print(f"Saved {self.num_gaussians} Gaussians to {output_path}")
        print("  View at: https://antimatter15.com/splat/")

    def _densify_and_prune(
        self,
        grad_threshold: Optional[float] = None,
        opacity_threshold: Optional[float] = None,
        max_screen_size: float = 20.0,
    ) -> Tuple[int, int]:
        """
        Adaptive density control: add and remove Gaussians.

        This method improves reconstruction quality by:
        1. Splitting/cloning Gaussians in under-reconstructed regions
           (high positional gradient indicates need for more detail)
        2. Removing Gaussians that contribute little
           (low opacity or very large scale)

        Args:
            grad_threshold: Gradient magnitude threshold for densification.
                If None, uses config.densify_grad_threshold.
            opacity_threshold: Minimum opacity to keep a Gaussian.
                If None, uses config.prune_opacity_threshold.
            max_screen_size: Maximum projected screen size before splitting.

        Returns:
            Tuple[int, int]: (num_added, num_removed)
                Number of Gaussians added and removed.

        Side Effects:
            - May increase or decrease total number of Gaussians
            - Updates all parameter tensors accordingly
            - Rebuilds optimizer with new parameters
            - Resets gradient accumulators

        Densification Strategy:
            For Gaussians with ||grad_position|| > threshold:
                - If scale > threshold: SPLIT into 2 smaller Gaussians
                - If scale < threshold: CLONE with slight position offset

        Pruning Strategy:
            Remove Gaussians where:
                - opacity < prune_opacity_threshold
                - scale > scene_extent (too large)

        Example:
            >>> added, removed = trainer._densify_and_prune()
            >>> print(f"Added {added}, removed {removed} Gaussians")
        """
        from .gaussian_utils import sigmoid

        if grad_threshold is None:
            grad_threshold = self.config.densify_grad_threshold
        if opacity_threshold is None:
            opacity_threshold = self.config.prune_opacity_threshold

        n_before = self.num_gaussians

        # Get current gradient magnitudes (averaged over accumulation period)
        if self._xyz_gradient_accum is None or self._denom is None:
            # No gradients accumulated yet
            return 0, 0

        # Compute average gradient magnitude
        grad_avg = self._xyz_gradient_accum / (self._denom + 1e-8)
        grad_mag = grad_avg.squeeze(-1)  # (N,)

        # Get actual scales (from log-space)
        scales_actual = torch.exp(self.scales.data)  # (N, 3)
        max_scale = scales_actual.max(dim=1).values  # (N,)

        # Compute scale percentile for split/clone decision
        scale_percentile_90 = torch.quantile(max_scale, 0.9)

        # =====================================================================
        # DENSIFICATION: Clone small Gaussians, Split large Gaussians
        # =====================================================================
        # The key insight from the 3DGS paper:
        # - High position gradient indicates the Gaussian is being "pulled"
        #   by the optimization, meaning the region needs more coverage.
        # - Small Gaussians with high gradient → CLONE (add more in that area)
        # - Large Gaussians with high gradient → SPLIT (break into smaller ones)
        # This adaptively increases resolution where needed.

        # Find Gaussians with high gradient (being pulled by optimization)
        high_grad_mask = grad_mag > grad_threshold

        # Clone: high gradient AND small scale (under-reconstruction)
        # These are fine-grained areas that need more Gaussians
        clone_mask = high_grad_mask & (max_scale <= scale_percentile_90)

        # Split: high gradient AND large scale (over-reconstruction)
        # These are blurry areas that need finer detail
        split_mask = high_grad_mask & (max_scale > scale_percentile_90)

        # Check max Gaussians limit
        n_to_add = clone_mask.sum().item() + split_mask.sum().item() * 2
        if self.num_gaussians + n_to_add > self.config.max_gaussians:
            # Reduce densification to stay within limits
            clone_mask = clone_mask & False  # Disable cloning
            split_mask = split_mask & False  # Disable splitting

        # === CLONE ===
        if clone_mask.any():
            clone_indices = clone_mask.nonzero(as_tuple=True)[0]

            # Create clones with slight position offset
            new_means = self.means.data[clone_indices].clone()
            offset = torch.randn_like(new_means) * 0.001
            new_means = new_means + offset

            new_scales = self.scales.data[clone_indices].clone()
            new_rotations = self.rotations.data[clone_indices].clone()
            new_sh_coeffs = self.sh_coeffs.data[clone_indices].clone()
            new_opacities = self.opacities.data[clone_indices].clone()

            # Append to existing
            self.means = nn.Parameter(torch.cat([self.means.data, new_means], dim=0))
            self.scales = nn.Parameter(torch.cat([self.scales.data, new_scales], dim=0))
            self.rotations = nn.Parameter(torch.cat([self.rotations.data, new_rotations], dim=0))
            self.sh_coeffs = nn.Parameter(torch.cat([self.sh_coeffs.data, new_sh_coeffs], dim=0))
            self.opacities = nn.Parameter(torch.cat([self.opacities.data, new_opacities], dim=0))

        # === SPLIT ===
        if split_mask.any():
            split_indices = split_mask.nonzero(as_tuple=True)[0]

            # Get Gaussians to split
            split_means = self.means.data[split_indices]
            split_scales = self.scales.data[split_indices]
            split_rotations = self.rotations.data[split_indices]
            split_sh_coeffs = self.sh_coeffs.data[split_indices]
            split_opacities = self.opacities.data[split_indices]

            # Create 2 new Gaussians per split Gaussian
            # Reduce scale by factor of 1.6 (approximately sqrt(2.56))
            # This ensures the combined volume of 2 new Gaussians roughly
            # equals the original. Since volume ~ scale^3, and we create 2:
            # 2 * (scale/1.6)^3 ≈ scale^3
            scale_reduction = torch.log(torch.tensor(1.6, device=self.device))

            # Offset in direction of largest scale
            scales_actual_split = torch.exp(split_scales)
            max_axis = scales_actual_split.argmax(dim=1)  # (N_split,)

            # Create offset vectors
            offset_mag = scales_actual_split.max(dim=1).values * 0.5  # (N_split,)
            offset1 = torch.zeros_like(split_means)
            offset2 = torch.zeros_like(split_means)

            for i in range(len(split_indices)):
                axis = max_axis[i].item()
                offset1[i, axis] = offset_mag[i]
                offset2[i, axis] = -offset_mag[i]

            # New positions
            new_means1 = split_means + offset1
            new_means2 = split_means + offset2

            # New scales (reduced)
            new_scales1 = split_scales - scale_reduction
            new_scales2 = split_scales - scale_reduction

            # Keep same rotation, color, reduce opacity slightly
            new_rotations1 = split_rotations.clone()
            new_rotations2 = split_rotations.clone()
            new_sh_coeffs1 = split_sh_coeffs.clone()
            new_sh_coeffs2 = split_sh_coeffs.clone()

            # Reduce opacity (in logit space, subtract to reduce)
            opacity_reduction = 0.5  # Reduce by ~half
            new_opacities1 = split_opacities - opacity_reduction
            new_opacities2 = split_opacities - opacity_reduction

            # Combine all new Gaussians
            new_means = torch.cat([new_means1, new_means2], dim=0)
            new_scales = torch.cat([new_scales1, new_scales2], dim=0)
            new_rotations = torch.cat([new_rotations1, new_rotations2], dim=0)
            new_sh_coeffs = torch.cat([new_sh_coeffs1, new_sh_coeffs2], dim=0)
            new_opacities = torch.cat([new_opacities1, new_opacities2], dim=0)

            # Remove original split Gaussians and add new ones
            keep_mask = ~split_mask
            self.means = nn.Parameter(torch.cat([self.means.data[keep_mask], new_means], dim=0))
            self.scales = nn.Parameter(torch.cat([self.scales.data[keep_mask], new_scales], dim=0))
            self.rotations = nn.Parameter(
                torch.cat([self.rotations.data[keep_mask], new_rotations], dim=0)
            )
            self.sh_coeffs = nn.Parameter(
                torch.cat([self.sh_coeffs.data[keep_mask], new_sh_coeffs], dim=0)
            )
            self.opacities = nn.Parameter(
                torch.cat([self.opacities.data[keep_mask], new_opacities], dim=0)
            )

        # =====================================================================
        # PRUNING: Remove low-opacity and too-large Gaussians
        # =====================================================================

        # Get current opacities (from logit-space)
        opacities_actual = sigmoid(self.opacities.data)  # (N,)

        # Prune low opacity
        prune_mask = opacities_actual < opacity_threshold

        # Also prune very large Gaussians (scene extent estimation)
        scales_actual = torch.exp(self.scales.data)
        max_scale = scales_actual.max(dim=1).values
        scene_extent = 2.0  # Approximate scene extent
        prune_mask = prune_mask | (max_scale > scene_extent)

        if prune_mask.any():
            keep_mask = ~prune_mask
            self.means = nn.Parameter(self.means.data[keep_mask])
            self.scales = nn.Parameter(self.scales.data[keep_mask])
            self.rotations = nn.Parameter(self.rotations.data[keep_mask])
            self.sh_coeffs = nn.Parameter(self.sh_coeffs.data[keep_mask])
            self.opacities = nn.Parameter(self.opacities.data[keep_mask])

        # =====================================================================
        # Rebuild optimizer and reset gradient accumulators
        # =====================================================================

        n_after = self.num_gaussians
        num_added = max(0, n_after - n_before + prune_mask.sum().item())
        num_removed = prune_mask.sum().item()

        # Reset gradient accumulators
        self._xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device=self.device)
        self._denom = torch.zeros((self.num_gaussians, 1), device=self.device)

        # Rebuild optimizer with new parameters
        self._rebuild_optimizer()

        return num_added, num_removed

    def _rebuild_optimizer(self) -> None:
        """Rebuild optimizer after densification/pruning changes parameter counts."""
        self.optimizer = torch.optim.Adam(
            [
                {"params": [self.means], "lr": 1.6e-4, "name": "means"},
                {"params": [self.scales], "lr": 5e-3, "name": "scales"},
                {"params": [self.rotations], "lr": 1e-3, "name": "rotations"},
                {"params": [self.sh_coeffs], "lr": 2.5e-3, "name": "sh_coeffs"},
                {"params": [self.opacities], "lr": 5e-2, "name": "opacities"},
            ]
        )

    def _accumulate_gradients(self) -> None:
        """Accumulate position gradients for densification decisions."""
        if self.means.grad is None:
            return

        # Accumulate gradient magnitude
        grad_mag = self.means.grad.norm(dim=1, keepdim=True)

        if self._xyz_gradient_accum is None:
            self._xyz_gradient_accum = torch.zeros_like(grad_mag)
            self._denom = torch.zeros_like(grad_mag)

        self._xyz_gradient_accum += grad_mag
        self._denom += 1

    @property
    def num_gaussians(self) -> int:
        """Return current number of Gaussian primitives."""
        if self.means is None:
            return 0
        return self.means.shape[0]

    def get_parameters(self) -> Dict[str, Tensor]:
        """
        Get all trainable Gaussian parameters.

        Returns:
            Dict[str, Tensor]: Dictionary containing:
                - 'means': (N, 3) positions
                - 'scales': (N, 3) log-scales
                - 'rotations': (N, 4) quaternions
                - 'sh_coeffs': (N, C, 3) SH coefficients
                - 'opacities': (N,) logit-opacities
        """
        return {
            "means": self.means,
            "scales": self.scales,
            "rotations": self.rotations,
            "sh_coeffs": self.sh_coeffs,
            "opacities": self.opacities,
        }
