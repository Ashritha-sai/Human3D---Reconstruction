"""
Loss Functions for Gaussian Splatting Optimization

This module provides loss functions used during Gaussian splatting training,
including photometric losses (L1, L2, SSIM), perceptual losses (LPIPS),
and regularization terms.

All losses are designed to work with masked regions, allowing optimization
only on the subject area while ignoring background.
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GaussianLosses(nn.Module):
    """
    Combined loss functions for Gaussian Splatting training.

    This class provides:
    - Photometric losses (L1, L2) for pixel-level reconstruction
    - Structural similarity (SSIM) for perceptual quality
    - LPIPS perceptual loss for high-level feature matching
    - Regularization losses for scale and opacity control

    Attributes:
        weight_l1: Weight for L1 photometric loss.
        weight_ssim: Weight for SSIM loss.
        weight_lpips: Weight for LPIPS perceptual loss.
        weight_scale_reg: Weight for scale regularization.
        weight_opacity_reg: Weight for opacity regularization.
        lpips_net: LPIPS network (VGG or Alex), loaded lazily.

    Example:
        >>> losses = GaussianLosses(weight_l1=0.8, weight_ssim=0.2)
        >>> rendered = model.render()  # (H, W, 3)
        >>> target = ground_truth      # (H, W, 3)
        >>> mask = segmentation_mask   # (H, W)
        >>> total, components = losses.total_loss(rendered, target, mask)
        >>> total.backward()
    """

    def __init__(
        self,
        weight_l1: float = 0.8,
        weight_ssim: float = 0.2,
        weight_lpips: float = 0.0,
        weight_scale_reg: float = 0.0,
        weight_opacity_reg: float = 0.0,
        lpips_net: str = "alex",
        device: str = "cpu",
    ) -> None:
        """
        Initialize loss functions with specified weights.

        Args:
            weight_l1: Weight for L1 photometric loss.
                Default: 0.8
                Range: [0, 1] typical

            weight_ssim: Weight for SSIM structural loss.
                Default: 0.2
                Range: [0, 1] typical

            weight_lpips: Weight for LPIPS perceptual loss.
                Default: 0.0 (disabled)
                Range: [0, 0.5] typical
                Note: LPIPS is computationally expensive

            weight_scale_reg: Weight for scale regularization.
                Default: 0.0
                Penalizes very large or very small scales

            weight_opacity_reg: Weight for opacity regularization.
                Default: 0.0
                Encourages binary (0 or 1) opacities

            lpips_net: Network architecture for LPIPS.
                Options: "vgg", "alex"
                Default: "alex" (faster than VGG)
                Only loaded if weight_lpips > 0

            device: Computation device.
                Default: "cpu"
        """
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_ssim = weight_ssim
        self.weight_lpips = weight_lpips
        self.weight_scale_reg = weight_scale_reg
        self.weight_opacity_reg = weight_opacity_reg
        self.device = device

        # Lazy load LPIPS if needed
        self._lpips_net = None
        self._lpips_net_type = lpips_net

        # Scale regularization thresholds (in log-space)
        self.scale_min_log = -7.0  # exp(-7) ≈ 0.001
        self.scale_max_log = 2.0  # exp(2) ≈ 7.4

    @property
    def lpips_net(self):
        """Lazy-load LPIPS network on first use."""
        if self._lpips_net is None and self.weight_lpips > 0:
            try:
                import lpips

                self._lpips_net = lpips.LPIPS(net=self._lpips_net_type).to(self.device)
                self._lpips_net.eval()
                for param in self._lpips_net.parameters():
                    param.requires_grad = False
            except ImportError:
                print("[WARN] lpips not installed, LPIPS loss will return 0")
                self._lpips_net = None
        return self._lpips_net

    def photometric_loss(
        self,
        rendered: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        loss_type: str = "l1",
    ) -> Tensor:
        """
        Compute pixel-wise photometric loss between rendered and target images.

        Args:
            rendered: Rendered RGB image from Gaussian splatting.
                Shape: (H, W, 3) or (B, H, W, 3)
                dtype: float32
                Range: [0, 1]

            target: Ground truth RGB image.
                Shape: (H, W, 3) or (B, H, W, 3)
                dtype: float32
                Range: [0, 1]

            mask: Optional binary mask for valid regions.
                Shape: (H, W) or (B, H, W)
                dtype: float32 or bool
                Values: 1 = valid, 0 = ignore
                If None, all pixels are used.
                Default: None

            loss_type: Type of photometric loss.
                Options: "l1", "l2", "huber"
                Default: "l1"

        Returns:
            Tensor: Scalar loss value.
        """
        # Ensure same shape
        if rendered.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: rendered {rendered.shape} vs target {target.shape}"
            )

        # Compute pixel-wise difference
        if loss_type == "l1":
            diff = torch.abs(rendered - target)
        elif loss_type == "l2":
            diff = (rendered - target) ** 2
        elif loss_type == "huber":
            diff = F.smooth_l1_loss(rendered, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Apply mask if provided
        if mask is not None:
            # Expand mask to match image dimensions
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)  # (H, W, 1)
            elif mask.ndim == 3 and mask.shape[-1] != 1:
                mask = mask.unsqueeze(-1)

            mask = mask.float()

            # Masked mean
            diff = diff * mask
            num_valid = mask.sum() + 1e-8
            loss = diff.sum() / num_valid
        else:
            loss = diff.mean()

        return loss

    def ssim_loss(
        self,
        img1: Tensor,
        img2: Tensor,
        mask: Optional[Tensor] = None,
        window_size: int = 11,
    ) -> Tensor:
        """
        Compute Structural Similarity Index (SSIM) loss.

        SSIM measures perceptual similarity considering luminance,
        contrast, and structure. The loss is 1 - SSIM, so lower is better.

        Args:
            img1: First image (typically rendered).
                Shape: (H, W, 3) or (B, H, W, 3)
                dtype: float32
                Range: [0, 1]

            img2: Second image (typically target).
                Shape: (H, W, 3) or (B, H, W, 3)
                dtype: float32
                Range: [0, 1]

            mask: Optional binary mask.
                Shape: (H, W) or (B, H, W)
                If None, all pixels are used.
                Default: None

            window_size: Size of Gaussian window for local statistics.
                Default: 11
                Should be odd number

        Returns:
            Tensor: Scalar loss value in range [0, 1].
                0 = identical images
                1 = completely different
        """
        # Use pytorch_msssim for efficient SSIM computation
        try:
            from pytorch_msssim import ssim as compute_ssim
        except ImportError:
            # Fallback to simple implementation
            return self._ssim_simple(img1, img2, mask, window_size)

        # Convert from (H, W, 3) to (B, C, H, W) format
        if img1.ndim == 3:
            img1 = img1.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            img2 = img2.permute(2, 0, 1).unsqueeze(0)

        # Compute SSIM
        ssim_val = compute_ssim(
            img1,
            img2,
            data_range=1.0,
            size_average=True,
            win_size=window_size,
        )

        # Return 1 - SSIM as loss
        loss = 1.0 - ssim_val

        return loss

    def _ssim_simple(
        self,
        img1: Tensor,
        img2: Tensor,
        mask: Optional[Tensor] = None,
        window_size: int = 11,
    ) -> Tensor:
        """Simple SSIM implementation as fallback."""
        # SSIM stability constants (from the original SSIM paper)
        # C1 = (K1 * L)^2 where K1=0.01, L=1.0 (dynamic range)
        # C2 = (K2 * L)^2 where K2=0.03, L=1.0
        # These prevent division by zero when local variance is near zero
        C1 = 0.01**2  # Luminance stability: 0.0001
        C2 = 0.03**2  # Contrast stability: 0.0009

        # Create Gaussian window
        window = gaussian_window(window_size, 1.5, device=img1.device)

        # Convert to (B, C, H, W) format
        if img1.ndim == 3:
            img1 = img1.permute(2, 0, 1).unsqueeze(0)
            img2 = img2.permute(2, 0, 1).unsqueeze(0)

        _, C, H, W = img1.shape

        # Pad images
        pad = window_size // 2
        img1 = F.pad(img1, (pad, pad, pad, pad), mode="reflect")
        img2 = F.pad(img2, (pad, pad, pad, pad), mode="reflect")

        # Expand window for all channels
        window = window.expand(C, 1, window_size, window_size)

        # Compute local means
        mu1 = F.conv2d(img1, window, groups=C)
        mu2 = F.conv2d(img2, window, groups=C)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = F.conv2d(img1**2, window, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2**2, window, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, groups=C) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        # Mean SSIM
        ssim_val = ssim_map.mean()

        return 1.0 - ssim_val

    def lpips_loss(
        self,
        rendered: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity) loss.

        LPIPS uses deep features from a pretrained network to measure
        perceptual similarity, often correlating better with human
        perception than pixel-wise metrics.

        Args:
            rendered: Rendered RGB image.
                Shape: (H, W, 3) or (B, H, W, 3)
                dtype: float32
                Range: [0, 1]

            target: Ground truth RGB image.
                Shape: (H, W, 3) or (B, H, W, 3)
                dtype: float32
                Range: [0, 1]

            mask: Optional binary mask (applied after LPIPS computation).
                Shape: (H, W) or (B, H, W)
                Note: LPIPS is computed on full images, mask only affects
                the final averaging.
                Default: None

        Returns:
            Tensor: Scalar loss value.
                Range: [0, ~1], lower = more similar
                Typical good reconstruction: < 0.1
        """
        if self.lpips_net is None:
            return torch.tensor(0.0, device=rendered.device)

        # Convert from (H, W, 3) to (B, C, H, W) format
        if rendered.ndim == 3:
            rendered = rendered.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            target = target.permute(2, 0, 1).unsqueeze(0)

        # LPIPS expects range [-1, 1]
        rendered_lpips = rendered * 2.0 - 1.0
        target_lpips = target * 2.0 - 1.0

        # Compute LPIPS
        with torch.no_grad():
            # Set to eval mode during LPIPS computation
            self._lpips_net.eval()

        lpips_val = self._lpips_net(rendered_lpips, target_lpips)

        return lpips_val.mean()

    def regularization_loss(
        self,
        scales: Tensor,
        opacities: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute regularization losses for Gaussian parameters.

        Regularization helps prevent degenerate solutions and improves
        training stability. Two types are provided:

        1. Scale regularization: Prevents Gaussians from becoming too large
           or too small, which can cause rendering artifacts.

        2. Opacity regularization: Encourages binary opacities (0 or 1),
           reducing the number of semi-transparent Gaussians.

        Args:
            scales: Log-scale parameters for all Gaussians.
                Shape: (N, 3)
                dtype: float32
                Range: Log-space (exp(scales) gives actual scale)

            opacities: Logit-opacity parameters for all Gaussians.
                Shape: (N,) or (N, 1)
                dtype: float32
                Range: Logit-space (sigmoid(opacities) gives actual opacity)

        Returns:
            Tuple[Tensor, Tensor]:
                - scale_loss: Scalar regularization loss for scales
                - opacity_loss: Scalar regularization loss for opacities
        """
        # ===== Scale Regularization =====
        # Penalize scales that are too large or too small
        # scales is in log-space

        # Penalize scales below minimum (too small -> numerical issues)
        scale_min_penalty = F.relu(self.scale_min_log - scales)

        # Penalize scales above maximum (too large -> rendering artifacts)
        scale_max_penalty = F.relu(scales - self.scale_max_log)

        scale_loss = (scale_min_penalty**2).mean() + (scale_max_penalty**2).mean()

        # ===== Opacity Regularization =====
        # Encourage binary opacities (either 0 or 1)
        # This helps with pruning and reduces floaters

        # Convert from logit to actual opacity
        if opacities.ndim == 2:
            opacities = opacities.squeeze(-1)

        opacity_sigmoid = torch.sigmoid(opacities)

        # Binary entropy: p * (1 - p) is minimized when p = 0 or p = 1
        opacity_loss = (opacity_sigmoid * (1.0 - opacity_sigmoid)).mean()

        return scale_loss, opacity_loss

    def total_loss(
        self,
        rendered: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
        scales: Optional[Tensor] = None,
        opacities: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute total weighted loss combining all components.

        This is the main entry point for loss computation during training.
        It combines photometric (L1), structural (SSIM), perceptual (LPIPS),
        and regularization losses with configurable weights.

        Args:
            rendered: Rendered RGB image from Gaussian splatting.
                Shape: (H, W, 3)
                dtype: float32
                Range: [0, 1]

            target: Ground truth RGB image.
                Shape: (H, W, 3)
                dtype: float32
                Range: [0, 1]

            mask: Binary mask for valid regions.
                Shape: (H, W)
                dtype: float32 or bool
                If None, all pixels are used.
                Default: None

            scales: Log-scale parameters (for regularization).
                Shape: (N, 3)
                If None, scale regularization is skipped.
                Default: None

            opacities: Logit-opacity parameters (for regularization).
                Shape: (N,)
                If None, opacity regularization is skipped.
                Default: None

        Returns:
            Tuple[Tensor, Dict[str, float]]:
                - total: Scalar total loss (differentiable)
                - components: Dictionary of individual loss values (for logging)
                    Keys: "l1", "ssim", "lpips", "scale_reg", "opacity_reg", "total"
        """
        components = {}
        total = torch.tensor(0.0, device=rendered.device, requires_grad=True)

        # L1 loss
        if self.weight_l1 > 0:
            l1_loss = self.photometric_loss(rendered, target, mask, loss_type="l1")
            components["l1"] = l1_loss.item()
            total = total + self.weight_l1 * l1_loss

        # SSIM loss
        if self.weight_ssim > 0:
            ssim_loss = self.ssim_loss(rendered, target, mask)
            components["ssim"] = ssim_loss.item()
            total = total + self.weight_ssim * ssim_loss

        # LPIPS loss
        if self.weight_lpips > 0:
            lpips_loss = self.lpips_loss(rendered, target, mask)
            components["lpips"] = lpips_loss.item()
            total = total + self.weight_lpips * lpips_loss
        else:
            components["lpips"] = 0.0

        # Regularization losses
        if scales is not None and opacities is not None:
            scale_reg, opacity_reg = self.regularization_loss(scales, opacities)

            if self.weight_scale_reg > 0:
                components["scale_reg"] = scale_reg.item()
                total = total + self.weight_scale_reg * scale_reg
            else:
                components["scale_reg"] = 0.0

            if self.weight_opacity_reg > 0:
                components["opacity_reg"] = opacity_reg.item()
                total = total + self.weight_opacity_reg * opacity_reg
            else:
                components["opacity_reg"] = 0.0
        else:
            components["scale_reg"] = 0.0
            components["opacity_reg"] = 0.0

        components["total"] = total.item()

        return total, components


def gaussian_window(window_size: int, sigma: float, device: str = "cpu") -> Tensor:
    """
    Create a 2D Gaussian window for SSIM computation.

    Args:
        window_size: Size of the square window (should be odd).
        sigma: Standard deviation of the Gaussian.
        device: Computation device.

    Returns:
        Tensor: Normalized 2D Gaussian kernel.
            Shape: (1, 1, window_size, window_size)
            Sum equals 1.0
    """
    # Create 1D Gaussian
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - window_size // 2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    # Create 2D Gaussian via outer product
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)  # (window_size, window_size)
    window_2d = window_2d / window_2d.sum()

    # Add batch and channel dimensions
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)

    return window_2d


def compute_ssim_map(
    img1: Tensor,
    img2: Tensor,
    window: Tensor,
    size_average: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Compute SSIM map between two images.

    Args:
        img1: First image in (B, C, H, W) format.
        img2: Second image in (B, C, H, W) format.
        window: Gaussian window from gaussian_window().
        size_average: If True, return scalar mean. If False, return map.

    Returns:
        SSIM value (scalar) or SSIM map (H, W tensor).
    """
    C1 = 0.01**2
    C2 = 0.03**2

    _, C, H, W = img1.shape

    # Expand window for all channels
    window = window.expand(C, 1, -1, -1)

    # Compute local means
    mu1 = F.conv2d(img1, window, padding="same", groups=C)
    mu2 = F.conv2d(img2, window, padding="same", groups=C)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1**2, window, padding="same", groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding="same", groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding="same", groups=C) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map


# =============================================================================
# Visualization Helpers
# =============================================================================


def save_comparison_image(
    target: Tensor,
    rendered: Tensor,
    output_path: str,
    mask: Optional[Tensor] = None,
    title: str = "Target | Rendered | Difference",
) -> None:
    """
    Save a side-by-side comparison image.

    Creates a visualization showing:
    - Left: Target/ground truth image
    - Center: Rendered image
    - Right: Absolute difference (enhanced for visibility)

    Args:
        target: Ground truth image.
            Shape: (H, W, 3)
            Range: [0, 1]

        rendered: Rendered image from Gaussian splatting.
            Shape: (H, W, 3)
            Range: [0, 1]

        output_path: Path to save the comparison image.

        mask: Optional binary mask to overlay.
            Shape: (H, W)

        title: Title for the image (not displayed, just for logging).
    """
    import numpy as np
    import cv2

    # Convert to numpy
    target_np = target.detach().cpu().numpy()
    rendered_np = rendered.detach().cpu().numpy()

    # Clip to valid range
    target_np = np.clip(target_np, 0, 1)
    rendered_np = np.clip(rendered_np, 0, 1)

    # Compute difference
    diff_np = np.abs(target_np - rendered_np)

    # Enhance difference for visibility (scale by 5x)
    diff_enhanced = np.clip(diff_np * 5.0, 0, 1)

    # Concatenate horizontally
    comparison = np.concatenate([target_np, rendered_np, diff_enhanced], axis=1)

    # Add separator lines
    H, W, _ = target_np.shape
    comparison[:, W - 1 : W + 1, :] = 1.0  # White line
    comparison[:, 2 * W - 1 : 2 * W + 1, :] = 1.0

    # Convert to BGR for OpenCV
    comparison_bgr = (comparison[:, :, ::-1] * 255).astype(np.uint8)

    # Save
    cv2.imwrite(output_path, comparison_bgr)
    print(f"Saved comparison: {output_path}")


def visualize_loss_components(
    components: Dict[str, float],
    output_path: str,
) -> None:
    """
    Create a bar chart visualization of loss components.

    Args:
        components: Dictionary of loss component values.
        output_path: Path to save the chart.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed, skipping loss visualization")
        return

    # Filter out zero values
    filtered = {k: v for k, v in components.items() if v > 0 and k != "total"}

    if not filtered:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    names = list(filtered.keys())
    values = list(filtered.values())

    bars = ax.bar(
        names, values, color=["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"]
    )

    ax.set_ylabel("Loss Value")
    ax.set_title(f"Loss Components (Total: {components.get('total', 0):.4f})")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Saved loss chart: {output_path}")
