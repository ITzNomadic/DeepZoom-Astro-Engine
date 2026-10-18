"""
AstroValidator V0.1.1 - Objective Scientific and Metric Evaluation Suite.
Implements standard local Gaussian-windowed SSIM (Wang et al., 2004),
control-referenced tile seam discontinuity analysis, and photometric metrics.
"""

from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import scipy.ndimage
from PIL import Image

class AstroValidator:
    def __init__(self, data_range: float = 255.0):
        self.data_range = float(data_range)

    def compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Computes Peak Signal-to-Noise Ratio (PSNR) in dB.
        """
        arr1 = img1.astype(np.float64)
        arr2 = img2.astype(np.float64)
        mse = float(np.mean((arr1 - arr2) ** 2))
        if mse <= 1e-12:
            return 100.0  # Identical bitwise match
        psnr = 10.0 * np.log10((self.data_range ** 2) / mse)
        return round(float(psnr), 2)

    def compute_ssim_channel(
        self,
        arr1: np.ndarray,
        arr2: np.ndarray,
        sigma: float = 1.5,
        radius: int = 5
    ) -> Tuple[float, np.ndarray]:
        """
        Computes standard local SSIM (Wang et al., 2004) on a single 2D channel.
        Window: (2*radius + 1)x(2*radius + 1) [default: 11x11] Gaussian kernel with standard deviation sigma=1.5.
        Constants: C1 = (0.01 * L)^2, C2 = (0.03 * L)^2.
        Padding: reflect boundary.
        """
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        f_kwargs = {'sigma': sigma, 'mode': 'reflect', 'radius': radius}
        
        mu1 = scipy.ndimage.gaussian_filter(arr1, **f_kwargs)
        mu2 = scipy.ndimage.gaussian_filter(arr2, **f_kwargs)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = scipy.ndimage.gaussian_filter(arr1 * arr1, **f_kwargs) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(arr2 * arr2, **f_kwargs) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(arr1 * arr2, **f_kwargs) - mu1_mu2

        # Enforce non-negative variance for numerical stability
        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)

        num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / den

        # Exclude outer boundary where convolution extends beyond image if large enough
        h, w = arr1.shape
        if h > 2 * radius and w > 2 * radius:
            valid_map = ssim_map[radius:-radius, radius:-radius]
        else:
            valid_map = ssim_map

        mssim = float(np.mean(valid_map))
        return mssim, ssim_map

    def compute_ssim(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        window_size: int = 11,
        sigma: float = 1.5
    ) -> float:
        """
        Computes standard local SSIM.
        For RGB images, evaluates luminance channel (Y in ITU-R BT.601) as primary standard.
        """
        res = self.compute_ssim_detailed(img1, img2, window_size=window_size, sigma=sigma)
        return res["ssim_y"]

    def compute_ssim_detailed(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        window_size: int = 11,
        sigma: float = 1.5
    ) -> Dict[str, Any]:
        """
        Computes standard local SSIM with full diagnostic parameters.
        Returns luminance SSIM, per-channel RGB SSIM, parameters and data range.
        """
        arr1 = img1.astype(np.float64)
        arr2 = img2.astype(np.float64)
        radius = window_size // 2

        if arr1.ndim == 3 and arr1.shape[2] >= 3:
            # 1. Luminance channel (ITU-R BT.601)
            y1 = 0.299 * arr1[..., 0] + 0.587 * arr1[..., 1] + 0.114 * arr1[..., 2]
            y2 = 0.299 * arr2[..., 0] + 0.587 * arr2[..., 1] + 0.114 * arr2[..., 2]
            ssim_y, _ = self.compute_ssim_channel(y1, y2, sigma=sigma, radius=radius)

            # 2. Per-channel RGB SSIM
            rgb_ssims = []
            for c in range(3):
                sc, _ = self.compute_ssim_channel(arr1[..., c], arr2[..., c], sigma=sigma, radius=radius)
                rgb_ssims.append(sc)
            ssim_rgb_mean = float(np.mean(rgb_ssims))
        else:
            gray1 = arr1 if arr1.ndim == 2 else arr1[..., 0]
            gray2 = arr2 if arr2.ndim == 2 else arr2[..., 0]
            ssim_y, _ = self.compute_ssim_channel(gray1, gray2, sigma=sigma, radius=radius)
            ssim_rgb_mean = ssim_y

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        return {
            "ssim_y": round(ssim_y, 4),
            "ssim_rgb": round(ssim_rgb_mean, 4),
            "window_size": window_size,
            "sigma": sigma,
            "c1": round(c1, 4),
            "c2": round(c2, 4),
            "data_range": self.data_range,
            "boundary_mode": "reflect",
            "evaluation": "valid_interior_crop"
        }

    def measure_tile_seams(
        self,
        image_rgb: np.ndarray,
        tile_size: int = 400,
        scale: int = 4,
        control_offset: int = 4
    ) -> Dict[str, Any]:
        """
        Scientifically measures potential stitching discontinuities along tile boundary lines.
        Compares boundary jump against adjacent non-seam control windows to eliminate false positives
        from legitimate astronomical gradients or structures.
        """
        arr = image_rgb.astype(np.float32)
        h, w = arr.shape[:2]
        stride = tile_size * scale

        seam_excess_list = []
        control_var_list = []
        inspected_lines = []

        # Check vertical seams
        for x in range(stride, w - 16, stride):
            if x - control_offset - 2 < 0 or x + control_offset + 2 >= w:
                continue

            # Step jump across seam boundary
            seam_jump = np.mean(np.abs(arr[:, x, :] - arr[:, x - 1, :]))

            # Control baseline jumps 4 pixels away on both sides
            left_ctrl = np.mean(np.abs(arr[:, x - control_offset, :] - arr[:, x - control_offset - 1, :]))
            right_ctrl = np.mean(np.abs(arr[:, x + control_offset, :] - arr[:, x + control_offset + 1, :]))
            ctrl_jump = 0.5 * (left_ctrl + right_ctrl)

            excess = float(seam_jump - ctrl_jump)
            seam_excess_list.append(excess)
            control_var_list.append(float(ctrl_jump))
            inspected_lines.append(f"V_{x}")

        # Check horizontal seams
        for y in range(stride, h - 16, stride):
            if y - control_offset - 2 < 0 or y + control_offset + 2 >= h:
                continue

            seam_jump = np.mean(np.abs(arr[y, :, :] - arr[y - 1, :, :]))
            top_ctrl = np.mean(np.abs(arr[y - control_offset, :, :] - arr[y - control_offset - 1, :, :]))
            bot_ctrl = np.mean(np.abs(arr[y + control_offset, :, :] - arr[y + control_offset + 1, :, :]))
            ctrl_jump = 0.5 * (top_ctrl + bot_ctrl)

            excess = float(seam_jump - ctrl_jump)
            seam_excess_list.append(excess)
            control_var_list.append(float(ctrl_jump))
            inspected_lines.append(f"H_{y}")

        if not seam_excess_list:
            return {
                "tested_tile_stride_px": stride,
                "total_seams_inspected": 0,
                "mean_seam_excess": 0.0,
                "max_seam_excess": 0.0,
                "mean_control_baseline": 0.0,
                "relative_seam_ratio": 1.0,
                "seam_blending_passed": True,
                "note": "Image smaller than single tile or boundary outside bounds"
            }

        mean_excess = float(np.mean(seam_excess_list))
        max_excess = float(np.max(seam_excess_list))
        mean_ctrl = float(np.mean(control_var_list))
        seam_ratio = float((mean_ctrl + max(mean_excess, 0.0)) / max(mean_ctrl, 1e-3))

        return {
            "tested_tile_stride_px": stride,
            "total_seams_inspected": len(seam_excess_list),
            "mean_seam_excess": round(mean_excess, 3),
            "max_seam_excess": round(max_excess, 3),
            "mean_control_baseline": round(mean_ctrl, 3),
            "relative_seam_ratio": round(seam_ratio, 3),
            "seam_blending_passed": bool(max_excess < 2.5 and seam_ratio < 1.35)
        }

    def evaluate_all(
        self,
        reference_img: np.ndarray,
        test_img: np.ndarray,
        tile_size: int = 400,
        scale: int = 4
    ) -> Dict[str, Any]:
        """
        Runs complete objective validation report between reference and test image.
        """
        psnr = self.compute_psnr(reference_img, test_img)
        ssim_res = self.compute_ssim_detailed(reference_img, test_img)
        seams = self.measure_tile_seams(test_img, tile_size=tile_size, scale=scale)

        diff = np.abs(reference_img.astype(np.float64) - test_img.astype(np.float64))
        mean_abs_diff = float(np.mean(diff))
        max_abs_diff = float(np.max(diff))
        
        # Spatial (2D pixel) modification tracking matching AdaptiveFuser
        if diff.ndim == 3:
            max_ch_diff = np.max(diff, axis=-1)
            modified_mask = max_ch_diff >= 0.5
            h, w = diff.shape[:2]
        else:
            max_ch_diff = diff
            modified_mask = diff >= 0.5
            h, w = diff.shape[:2]

        modified_pixels = int(np.sum(modified_mask))
        total_pixels = int(h * w)
        modified_pct = float(modified_pixels / total_pixels * 100.0)

        # Percentiles over modified pixels
        if modified_pixels > 0:
            mod_diffs = max_ch_diff[modified_mask]
            p50 = float(np.percentile(mod_diffs, 50.0))
            p90 = float(np.percentile(mod_diffs, 90.0))
            p99 = float(np.percentile(mod_diffs, 99.0))
        else:
            p50 = p90 = p99 = 0.0

        ssim_score = ssim_res["ssim_y"]
        if ssim_score >= 0.98 and psnr >= 40.0:
            quality = "A+"
        elif ssim_score >= 0.93:
            quality = "A"
        elif ssim_score >= 0.85:
            quality = "B"
        else:
            quality = "C"

        return {
            "psnr_db": psnr,
            "ssim": ssim_score,
            "ssim_rgb": ssim_res["ssim_rgb"],
            "ssim_details": ssim_res,
            "tiling_seams": seams,
            "spatial_delta": {
                "mean_absolute_difference": round(mean_abs_diff, 4),
                "max_absolute_difference": round(max_abs_diff, 2),
                "modified_pixel_count": modified_pixels,
                "modified_pixel_pct": round(modified_pct, 3),
                "p50_difference": round(p50, 2),
                "p90_difference": round(p90, 2),
                "p99_difference": round(p99, 2)
            },
            "quality_grade": quality
        }
