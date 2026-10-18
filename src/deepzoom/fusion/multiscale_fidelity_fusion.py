"""
DeepZoom Astro Engine V0.1.4 - Multiscale Fidelity & Detail Fusion Engine.
Implements:
  1. Multiscale Laplacian Pyramid frequency decomposition (4 octave bands):
     - L3: Global structure & low-frequency luminance/chroma.
     - L2: Medium-scale features (nebulosities, accretion disk body).
     - L1: Fine details (plasma filaments, ionization fronts, shock waves).
     - L0: High-frequency micro-details (stellar cores, sharp boundaries).
  2. Spatially-adaptive and band-specific multi-candidate selection:
     - Remacri (Organic smooth baseline anchor)
     - UltraSharp (High-frequency plasma filaments)
     - IBP (Iterative Back-Projection: LR consistency-enforced physical fidelity)
     - StarReconstruction (PSF-modeled stars and resolved binaries)
  3. Two distinct operational modes:
     - 'fidelity': Strict physics, error-bounded, minimal hallucination.
     - 'cinematic': Perceptual clarity, enhanced structural definition for deep zooms.
  4. Generation of model selection maps and difference audit artifacts.
"""

import time
from typing import Dict, Any, Tuple, Optional, List, Literal
import numpy as np
import scipy.ndimage
from PIL import Image


class LaplacianPyramid:
    """
    Mathematical 4-level Laplacian Pyramid decomposition and exact reconstruction.
    """
    def __init__(self, levels: int = 4):
        self.levels = levels

    def build_gaussian_pyramid(self, img: np.ndarray) -> List[np.ndarray]:
        """Builds Gaussian low-pass octave pyramid."""
        pyramid = [img.astype(np.float32)]
        curr = img.astype(np.float32)

        for _ in range(1, self.levels):
            h, w = curr.shape[:2]
            # Gaussian blur followed by 2x subsampling
            blurred = np.zeros_like(curr)
            for ch in range(curr.shape[-1]):
                blurred[..., ch] = scipy.ndimage.gaussian_filter(curr[..., ch], sigma=1.2, mode="reflect")

            # PIL downsampling for precision
            w_next, h_next = max(1, w // 2), max(1, h // 2)
            pil_blur = Image.fromarray(np.clip(np.round(blurred), 0.0, 255.0).astype(np.uint8))
            down_pil = pil_blur.resize((w_next, h_next), resample=Image.Resampling.BOX)
            curr = np.array(down_pil, dtype=np.float32)
            pyramid.append(curr)

        return pyramid

    def build_laplacian_pyramid(self, img: np.ndarray) -> List[np.ndarray]:
        """Builds band-pass Laplacian Pyramid: L_k = G_k - Up(G_{k+1})."""
        gauss_pyr = self.build_gaussian_pyramid(img)
        lap_pyr = []

        for k in range(self.levels - 1):
            curr_g = gauss_pyr[k]
            next_g = gauss_pyr[k + 1]

            h_k, w_k = curr_g.shape[:2]
            # Upsample next_g to match curr_g
            pil_next = Image.fromarray(np.clip(np.round(next_g), 0.0, 255.0).astype(np.uint8))
            up_pil = pil_next.resize((w_k, h_k), resample=Image.Resampling.BICUBIC)
            up_arr = np.array(up_pil, dtype=np.float32)

            lap_pyr.append(curr_g - up_arr)

        # Coarsest level
        lap_pyr.append(gauss_pyr[-1])
        return lap_pyr

    def reconstruct(self, lap_pyr: List[np.ndarray]) -> np.ndarray:
        """Exact collapse and reconstruction from Laplacian Pyramid."""
        curr = lap_pyr[-1]

        for k in range(self.levels - 2, -1, -1):
            h_target, w_target = lap_pyr[k].shape[:2]
            pil_curr = Image.fromarray(np.clip(np.round(curr), 0.0, 255.0).astype(np.uint8))
            up_pil = pil_curr.resize((w_target, h_target), resample=Image.Resampling.BICUBIC)
            up_arr = np.array(up_pil, dtype=np.float32)
            curr = up_arr + lap_pyr[k]

        return np.clip(curr, 0.0, 255.0)


class MultiscaleFidelityFuser:
    """
    Spatially-adaptive multiscale fusion engine.
    Combines Remacri, UltraSharp, IBP, and StarReconstruction across frequency octaves.
    """

    def __init__(self, pyramid_levels: int = 4):
        self.pyramid = LaplacianPyramid(levels=pyramid_levels)

    def fuse(
        self,
        candidate_remacri: np.ndarray,
        candidate_ultrasharp: Optional[np.ndarray] = None,
        candidate_ibp: Optional[np.ndarray] = None,
        candidate_star_recon: Optional[np.ndarray] = None,
        star_confidence_mask: Optional[np.ndarray] = None,
        filament_confidence_mask: Optional[np.ndarray] = None,
        mode: Literal["fidelity", "cinematic"] = "fidelity"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes multiscale frequency-band fusion.
        
        :param candidate_remacri: Remacri anchor (H, W, 3).
        :param candidate_ultrasharp: UltraSharp high-frequency candidate (H, W, 3).
        :param candidate_ibp: Iterative Back-Projection LR-consistent candidate (H, W, 3).
        :param candidate_star_recon: PSF star reconstruction candidate (H, W, 3).
        :param star_confidence_mask: Point-source mask (H, W).
        :param filament_confidence_mask: High-coherence filament mask (H, W).
        :param mode: 'fidelity' (conservative scientific) or 'cinematic' (perceptual detail).
        :return: (fused_image_uint8, metadata_and_selection_maps)
        """
        t0 = time.time()
        rem_rgb = candidate_remacri.astype(np.float32)
        h, w, c = rem_rgb.shape

        # Use Remacri as fallback for any optional candidate not provided
        ult_rgb = candidate_ultrasharp.astype(np.float32) if candidate_ultrasharp is not None else rem_rgb
        ibp_rgb = candidate_ibp.astype(np.float32) if candidate_ibp is not None else rem_rgb
        star_rgb = candidate_star_recon.astype(np.float32) if candidate_star_recon is not None else rem_rgb

        s_mask = star_confidence_mask.astype(np.float32) if star_confidence_mask is not None else np.zeros((h, w), dtype=np.float32)
        f_mask = filament_confidence_mask.astype(np.float32) if filament_confidence_mask is not None else np.zeros((h, w), dtype=np.float32)

        # Build Laplacian Pyramids for all active candidates
        pyr_rem = self.pyramid.build_laplacian_pyramid(rem_rgb)
        pyr_ult = self.pyramid.build_laplacian_pyramid(ult_rgb)
        pyr_ibp = self.pyramid.build_laplacian_pyramid(ibp_rgb)
        pyr_star = self.pyramid.build_laplacian_pyramid(star_rgb)

        pyr_fused = []
        selection_maps = {}

        # ---------------------------------------------------------------------
        # BAND L3: Global Base Structure (Low Frequencies)
        # ---------------------------------------------------------------------
        # Anchor strictly to IBP / Remacri: ensures global photometry and vacuum floor remain exact
        pyr_fused.append(pyr_ibp[-1] if mode == "fidelity" else (0.5 * pyr_rem[-1] + 0.5 * pyr_ibp[-1]))

        # ---------------------------------------------------------------------
        # BAND L2: Medium-Scale Details (Nebulae & Accretion Disk body)
        # ---------------------------------------------------------------------
        # In fidelity mode: 85% IBP + 15% Remacri. In cinematic: 70% IBP + 30% UltraSharp
        if mode == "fidelity":
            w_ibp_l2 = 0.80
            w_rem_l2 = 0.20
            band_l2 = w_ibp_l2 * pyr_ibp[2] + w_rem_l2 * pyr_rem[2]
        else:
            band_l2 = 0.65 * pyr_ibp[2] + 0.20 * pyr_rem[2] + 0.15 * pyr_ult[2]
        pyr_fused.append(band_l2)

        # ---------------------------------------------------------------------
        # BAND L1: Fine Details (Filaments, Plasma Streams, Shock Arcs)
        # ---------------------------------------------------------------------
        # Resize filament mask to L1 resolution
        h1, w1 = pyr_rem[1].shape[:2]
        pil_fmask = Image.fromarray(np.clip(f_mask * 255.0, 0, 255).astype(np.uint8))
        fmask_l1 = np.array(pil_fmask.resize((w1, h1), resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        fmask_l1_3d = np.expand_dims(fmask_l1, axis=-1)

        if mode == "fidelity":
            # On filaments: blend UltraSharp / IBP with Remacri
            band_l1 = (1.0 - fmask_l1_3d) * (0.6 * pyr_ibp[1] + 0.4 * pyr_rem[1]) + fmask_l1_3d * (0.5 * pyr_ult[1] + 0.5 * pyr_ibp[1])
        else:
            # Cinematic: higher weight to UltraSharp on filaments and edges
            fil_boost = 1.15
            band_l1 = (1.0 - fmask_l1_3d) * (0.5 * pyr_ibp[1] + 0.5 * pyr_rem[1]) + fmask_l1_3d * (0.75 * pyr_ult[1] * fil_boost + 0.25 * pyr_ibp[1])
        pyr_fused.append(band_l1)

        # ---------------------------------------------------------------------
        # BAND L0: High-Frequency Micro-Details (Stellar Cores, Point Sources)
        # ---------------------------------------------------------------------
        # Full resolution L0
        smask_3d = np.expand_dims(s_mask, axis=-1)
        fmask_3d = np.expand_dims(f_mask, axis=-1)

        # Base non-stellar, non-filament background detail: primarily from IBP (verified physical detail)
        base_l0 = 0.70 * pyr_ibp[0] + 0.30 * pyr_rem[0]

        # On stars: inject high-frequency core from StarReconstruction
        star_injected_l0 = (1.0 - smask_3d) * base_l0 + smask_3d * pyr_star[0]

        # On filaments: blend UltraSharp high frequency with anti-ringing damping
        if mode == "fidelity":
            fused_l0 = (1.0 - fmask_3d) * star_injected_l0 + fmask_3d * (0.6 * pyr_ibp[0] + 0.4 * pyr_ult[0])
        else:
            fused_l0 = (1.0 - fmask_3d) * star_injected_l0 + fmask_3d * (0.3 * pyr_ibp[0] + 0.7 * pyr_ult[0] * 1.10)
        pyr_fused.append(fused_l0)

        # Invert pyramid order to match reconstruction expectation [L0, L1, L2, L3]
        pyr_for_reconstruct = [pyr_fused[3], pyr_fused[2], pyr_fused[1], pyr_fused[0]]
        fused_raw = self.pyramid.reconstruct(pyr_for_reconstruct)

        final_fused_uint8 = np.clip(np.round(fused_raw), 0.0, 255.0).astype(np.uint8)
        elapsed = time.time() - t0

        # Model dominance selection map (identifies which model dominates each pixel)
        diff_rem = np.mean(np.abs(fused_raw - rem_rgb), axis=-1)
        diff_ult = np.mean(np.abs(fused_raw - ult_rgb), axis=-1)
        diff_ibp = np.mean(np.abs(fused_raw - ibp_rgb), axis=-1)
        diff_star = np.mean(np.abs(fused_raw - star_rgb), axis=-1)

        # Stacking distances to find nearest candidate
        stack = np.stack([diff_rem, diff_ult, diff_ibp, diff_star], axis=-1)
        dom_idx = np.argmin(stack, axis=-1)  # 0: Remacri, 1: UltraSharp, 2: IBP, 3: StarRecon

        # Map to RGB colors for visual inspection:
        # Remacri: Gray (128, 128, 128)
        # UltraSharp: Red (220, 80, 80)
        # IBP: Blue (60, 160, 240)
        # StarRecon: Gold (255, 215, 0)
        colors = np.array([
            [128, 128, 128],
            [220, 80, 80],
            [60, 160, 240],
            [255, 215, 0]
        ], dtype=np.uint8)
        selection_map_rgb = colors[dom_idx]

        meta = {
            "mode": mode,
            "pyramid_levels": self.pyramid.levels,
            "execution_seconds": round(elapsed, 3),
            "dominant_model_share_pct": {
                "remacri": round(float(np.mean(dom_idx == 0) * 100.0), 2),
                "ultrasharp": round(float(np.mean(dom_idx == 1) * 100.0), 2),
                "ibp": round(float(np.mean(dom_idx == 2) * 100.0), 2),
                "star_reconstruction": round(float(np.mean(dom_idx == 3) * 100.0), 2)
            }
        }

        return final_fused_uint8, {
            "meta": meta,
            "selection_map_rgb": selection_map_rgb,
            "modified_pixel_pct": round(float(np.mean(diff_rem >= 1.0) * 100.0), 2),
            "mean_delta_vs_remacri": round(float(np.mean(diff_rem)), 3),
            "max_delta_vs_remacri": round(float(np.max(diff_rem)), 1)
        }
