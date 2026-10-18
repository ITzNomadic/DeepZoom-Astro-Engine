"""
DeepZoom Astro Engine V0.1.4 - Iterative Back-Projection (IBP) Engine.
Enforces physical reconstruction consistency: D(HR) ~= LR_original.
Implements:
  1. Configurable forward degradation operators D(HR):
     - Area / Box integration (CCD sensor physical pixel binning)
     - Gaussian PSF blur + decimation (optical atmospheric/lens PSF)
     - Bicubic downsampling (standard computer vision degradation)
  2. Regularized iterative error back-projection:
     HR_{k+1} = HR_k + alpha * Up(LR - D(HR_k)) - lambda * TV(HR_k)
  3. Gibbs ringing suppression and edge-preserving damping.
  4. Multi-channel color preservation and numerical stability checks.
"""

import time
from typing import Dict, Any, Tuple, Optional, Literal
import numpy as np
import scipy.ndimage
from PIL import Image


class IterativeBackProjector:
    """
    Implements iterative back-projection to constrain super-resolved images
    to be mathematically consistent with the low-resolution observations.
    """

    def __init__(
        self,
        default_alpha: float = 0.45,
        default_tv_reg: float = 0.005,
        default_iterations: int = 10
    ):
        self.default_alpha = default_alpha
        self.default_tv_reg = default_tv_reg
        self.default_iterations = default_iterations

    def forward_degrade(
        self,
        hr_image: np.ndarray,
        target_shape: Tuple[int, int],
        method: Literal["box", "gaussian", "bicubic"] = "box",
        psf_sigma: float = 0.6
    ) -> np.ndarray:
        """
        Simulates forward camera / telescope degradation operator D(HR).
        :param hr_image: HR image array (H_hr, W_hr, C) in float32.
        :param target_shape: Target LR dimensions (H_lr, W_lr).
        :param method: Degradation mode.
        :param psf_sigma: PSF blur width before sampling.
        :return: Simulated LR image (H_lr, W_lr, C).
        """
        arr = hr_image.astype(np.float32)
        h_lr, w_lr = target_shape
        h_hr, w_hr, c = arr.shape

        if method == "gaussian" and psf_sigma > 0.0:
            blurred = np.zeros_like(arr)
            for ch in range(c):
                blurred[..., ch] = scipy.ndimage.gaussian_filter(arr[..., ch], sigma=psf_sigma, mode="reflect")
            arr = blurred

        # Convert to PIL for precision area/bicubic resampling
        pil_im = Image.fromarray(np.clip(np.round(arr), 0.0, 255.0).astype(np.uint8))
        if method == "box":
            lr_pil = pil_im.resize((w_lr, h_lr), resample=Image.Resampling.BOX)
        elif method == "bicubic":
            lr_pil = pil_im.resize((w_lr, h_lr), resample=Image.Resampling.BICUBIC)
        else:  # Lanczos / Gaussian
            lr_pil = pil_im.resize((w_lr, h_lr), resample=Image.Resampling.LANCZOS)

        return np.array(lr_pil, dtype=np.float32)

    def upsample_residual(
        self,
        residual_lr: np.ndarray,
        target_shape: Tuple[int, int],
        method: str = "lanczos"
    ) -> np.ndarray:
        """
        Upsamples residual error back into HR spatial domain.
        """
        h_hr, w_hr = target_shape
        c = residual_lr.shape[-1]
        res_hr = np.zeros((h_hr, w_hr, c), dtype=np.float32)

        # Scale and offset to preserve signed differences through PIL
        scale_val = 4.0
        offset_val = 128.0
        norm_lr = np.clip(residual_lr * scale_val + offset_val, 0.0, 255.0).astype(np.uint8)
        norm_pil = Image.fromarray(norm_lr)

        res_mode = Image.Resampling.LANCZOS if method == "lanczos" else Image.Resampling.BICUBIC
        up_pil = norm_pil.resize((w_hr, h_hr), resample=res_mode)
        up_arr = np.array(up_pil, dtype=np.float32)

        res_hr = (up_arr - offset_val) / scale_val
        return res_hr

    def compute_tv_gradient(self, img: np.ndarray) -> np.ndarray:
        """
        Computes Total Variation gradient for edge-preserving regularization.
        Suppresses noise and high-frequency ringing buildup.
        """
        eps = 1e-3
        gy, gx = np.gradient(img, axis=(0, 1))
        norm = np.sqrt(gx ** 2 + gy ** 2 + eps ** 2)
        div_x = np.gradient(gx / norm, axis=1)
        div_y = np.gradient(gy / norm, axis=0)
        return -(div_x + div_y)

    def run(
        self,
        hr_initial: np.ndarray,
        lr_original: np.ndarray,
        scale: int = 4,
        iterations: Optional[int] = None,
        alpha: Optional[float] = None,
        tv_weight: Optional[float] = None,
        degradation_method: Literal["box", "gaussian", "bicubic"] = "box",
        psf_sigma: float = 0.6
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes regularized Iterative Back-Projection.
        
        :param hr_initial: Initial high-resolution image (e.g. from Remacri) as uint8 or float32.
        :param lr_original: Original low-resolution input image as uint8 or float32.
        :param scale: Scale factor (default: 4).
        :param iterations: Number of iterations (e.g. 5, 10, 20).
        :param alpha: Update step size (default: 0.45).
        :param tv_weight: Total variation regularizer weight to prevent ringing (default: 0.005).
        :param degradation_method: Forward model degradation type.
        :param psf_sigma: PSF blur parameter.
        :return: (reconstructed_hr_uint8, metadata_dict)
        """
        t0 = time.time()
        n_iter = iterations if iterations is not None else self.default_iterations
        step_alpha = alpha if alpha is not None else self.default_alpha
        reg_lambda = tv_weight if tv_weight is not None else self.default_tv_reg

        hr_curr = hr_initial.astype(np.float32).copy()
        lr_orig = lr_original.astype(np.float32)

        h_hr, w_hr, c = hr_curr.shape
        h_lr, w_lr = lr_orig.shape[:2]

        iteration_history = []
        initial_error = None

        for it in range(n_iter):
            # 1. Forward degradation of current HR estimate
            lr_sim = self.forward_degrade(
                hr_image=hr_curr,
                target_shape=(h_lr, w_lr),
                method=degradation_method,
                psf_sigma=psf_sigma
            )

            # 2. Compute residual in LR domain
            residual_lr = lr_orig - lr_sim
            mae_lr = float(np.mean(np.abs(residual_lr)))
            rmse_lr = float(np.sqrt(np.mean(residual_lr ** 2)))

            if it == 0:
                initial_error = {"mae": round(mae_lr, 3), "rmse": round(rmse_lr, 3)}

            iteration_history.append({
                "iteration": it + 1,
                "lr_mae": round(mae_lr, 4),
                "lr_rmse": round(rmse_lr, 4)
            })

            # 3. Check convergence
            if mae_lr < 0.20:
                break

            # 4. Upsample residual to HR domain
            residual_hr = self.upsample_residual(residual_lr, (h_hr, w_hr))

            # 5. Regularization (TV dampening to avoid ringing accumulation)
            if reg_lambda > 0.0:
                tv_grad = np.zeros_like(hr_curr)
                for ch in range(c):
                    tv_grad[..., ch] = self.compute_tv_gradient(hr_curr[..., ch])
                update = step_alpha * residual_hr - reg_lambda * tv_grad
            else:
                update = step_alpha * residual_hr

            # 6. Apply update and clip
            hr_curr = np.clip(hr_curr + update, 0.0, 255.0)

        # Final consistency check
        lr_final_sim = self.forward_degrade(
            hr_image=hr_curr,
            target_shape=(h_lr, w_lr),
            method=degradation_method,
            psf_sigma=psf_sigma
        )
        final_mae = float(np.mean(np.abs(lr_orig - lr_final_sim)))
        final_rmse = float(np.sqrt(np.mean((lr_orig - lr_final_sim) ** 2)))

        elapsed = time.time() - t0
        final_hr_uint8 = np.clip(np.round(hr_curr), 0.0, 255.0).astype(np.uint8)

        meta = {
            "method": "iterative_back_projection",
            "iterations_completed": len(iteration_history),
            "step_alpha": step_alpha,
            "tv_weight": reg_lambda,
            "degradation_method": degradation_method,
            "initial_lr_error": initial_error,
            "final_lr_error": {"mae": round(final_mae, 4), "rmse": round(final_rmse, 4)},
            "error_reduction_pct": round((1.0 - (final_mae / max(initial_error["mae"], 1e-4))) * 100.0, 2),
            "execution_seconds": round(elapsed, 3),
            "history": iteration_history
        }

        return final_hr_uint8, meta
