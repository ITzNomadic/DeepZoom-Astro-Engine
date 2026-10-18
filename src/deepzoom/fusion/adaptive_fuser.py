"""
Adaptive Fusion Layer V0.1.2 for DeepZoom Studio.
Blends frozen Remacri baseline with StarPreserve, FilamentGuard, and DarkSky protections.
Strictly implements:
1. Single-Weight Contract: blend_weight applied strictly ONCE (zero double attenuation).
2. Pixel modification metrics synchronization: candidate vs effectively modified (|delta| >= 1).
3. Percentile statistics: p50, p90, p99, mean, max difference.
4. Difference maps generation and export (individual layer diffs, total diff, amplified 15x).
5. Untouched pixels remain bitwise identical to baseline.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Literal, Union
import numpy as np
from PIL import Image

AblationMode = Literal[
    "baseline",
    "star_preserve_only",
    "filament_guard_only",
    "dark_sky_only",
    "star_plus_filament",
    "all_layers"
]


class AdaptiveFuser:
    def __init__(self, default_weight: float = 0.5):
        self.default_weight = float(np.clip(default_weight, 0.0, 1.0))

    def fuse(
        self,
        baseline_rgb: np.ndarray,
        mode: AblationMode = "all_layers",
        star_reconstructed_rgb: Optional[np.ndarray] = None,
        star_mask: Optional[np.ndarray] = None,
        filament_repaired_rgb: Optional[np.ndarray] = None,
        filament_mask: Optional[np.ndarray] = None,
        dark_sky_rgb: Optional[np.ndarray] = None,
        dark_sky_mask: Optional[np.ndarray] = None,
        weight: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adaptively fuses baseline with astronomical protection layers based on the requested ablation mode.
        Conforms strictly to the Single-Weight Contract.
        
        Returns:
            fused_rgb: uint8 array
            stats: detailed dictionary of spatial modifications and delta statistics
        """
        w = self.default_weight if weight is None else float(np.clip(weight, 0.0, 1.0))

        h, w_img, c = baseline_rgb.shape
        total_pixels = h * w_img

        if mode == "baseline" or w <= 0.0:
            return baseline_rgb.copy(), {
                "mode": "baseline",
                "blend_weight": 0.0,
                "candidate_pixels": 0,
                "candidate_pct": 0.0,
                "effectively_modified_pixels": 0,
                "effectively_modified_pct": 0.0,
                "mean_absolute_difference": 0.0,
                "max_absolute_difference": 0.0,
                "p50_difference": 0.0,
                "p90_difference": 0.0,
                "p99_difference": 0.0,
                "active_layers": [],
                "per_layer_stats": {}
            }

        base = baseline_rgb.astype(np.float32)
        accum_delta = np.zeros_like(base)
        active_layers = []
        per_layer_stats = {}

        # 1. StarPreserve layer (Single-Weight: w applied once)
        star_layer_delta = np.zeros_like(base)
        if mode in ("star_preserve_only", "star_plus_filament", "all_layers"):
            if star_reconstructed_rgb is not None and star_mask is not None:
                pure_star_delta = star_reconstructed_rgb.astype(np.float32) - base
                s_mask = np.expand_dims(np.clip(star_mask, 0.0, 1.0), axis=-1)
                star_layer_delta = w * s_mask * pure_star_delta
                accum_delta += star_layer_delta
                active_layers.append("StarPreserve")
                per_layer_stats["StarPreserve"] = self._compute_delta_stats(star_layer_delta, star_mask)

        # 2. FilamentGuard layer (Single-Weight: w applied once)
        filament_layer_delta = np.zeros_like(base)
        if mode in ("filament_guard_only", "star_plus_filament", "all_layers"):
            if filament_repaired_rgb is not None and filament_mask is not None:
                pure_fil_delta = filament_repaired_rgb.astype(np.float32) - base
                f_mask = np.expand_dims(np.clip(filament_mask, 0.0, 1.0), axis=-1)
                filament_layer_delta = w * f_mask * pure_fil_delta
                accum_delta += filament_layer_delta
                active_layers.append("FilamentGuard")
                per_layer_stats["FilamentGuard"] = self._compute_delta_stats(filament_layer_delta, filament_mask)

        # 3. DarkSky layer (Single-Weight: w applied once)
        dark_layer_delta = np.zeros_like(base)
        if mode in ("dark_sky_only", "all_layers"):
            if dark_sky_rgb is not None and dark_sky_mask is not None:
                pure_dark_delta = dark_sky_rgb.astype(np.float32) - base
                d_mask = np.expand_dims(np.clip(dark_sky_mask, 0.0, 1.0), axis=-1)
                dark_layer_delta = w * d_mask * pure_dark_delta
                accum_delta += dark_layer_delta
                active_layers.append("DarkSky")
                per_layer_stats["DarkSky"] = self._compute_delta_stats(dark_layer_delta, dark_sky_mask)

        output = base + accum_delta
        final_rgb = np.clip(np.round(output), 0.0, 255.0).astype(np.uint8)

        # Unified pixel modification statistics
        abs_diff = np.abs(final_rgb.astype(np.int32) - baseline_rgb.astype(np.int32))
        max_ch_diff = np.max(abs_diff, axis=-1)
        effectively_modified_mask = max_ch_diff >= 1
        effective_count = int(np.sum(effectively_modified_mask))
        effective_pct = float(effective_count / total_pixels * 100.0)

        # Candidate pixels: where any active mask was positive (> 0.02)
        combined_candidate_mask = np.zeros((h, w_img), dtype=bool)
        if "StarPreserve" in active_layers and star_mask is not None:
            combined_candidate_mask |= (star_mask > 0.02)
        if "FilamentGuard" in active_layers and filament_mask is not None:
            combined_candidate_mask |= (filament_mask > 0.02)
        if "DarkSky" in active_layers and dark_sky_mask is not None:
            combined_candidate_mask |= (dark_sky_mask > 0.02)
        candidate_count = int(np.sum(combined_candidate_mask))
        candidate_pct = float(candidate_count / total_pixels * 100.0)

        mean_abs_diff = float(np.mean(abs_diff))
        max_abs_diff = float(np.max(abs_diff))

        # Percentiles over effectively modified pixels
        if effective_count > 0:
            mod_diffs = max_ch_diff[effectively_modified_mask]
            p50 = float(np.percentile(mod_diffs, 50.0))
            p90 = float(np.percentile(mod_diffs, 90.0))
            p99 = float(np.percentile(mod_diffs, 99.0))
        else:
            p50 = p90 = p99 = 0.0

        return final_rgb, {
            "mode": mode,
            "blend_weight": w,
            "candidate_pixels": candidate_count,
            "candidate_pct": round(candidate_pct, 3),
            "effectively_modified_pixels": effective_count,
            "effectively_modified_pct": round(effective_pct, 3),
            "mean_absolute_difference": round(mean_abs_diff, 4),
            "max_absolute_difference": round(max_abs_diff, 2),
            "p50_difference": round(p50, 2),
            "p90_difference": round(p90, 2),
            "p99_difference": round(p99, 2),
            "active_layers": active_layers,
            "per_layer_stats": per_layer_stats
        }

    def generate_difference_maps(
        self,
        baseline_rgb: np.ndarray,
        fused_rgb: np.ndarray,
        star_reconstructed_rgb: Optional[np.ndarray] = None,
        filament_repaired_rgb: Optional[np.ndarray] = None,
        dark_sky_rgb: Optional[np.ndarray] = None,
        amplification_factor: float = 15.0
    ) -> Dict[str, np.ndarray]:
        """
        Generates 2D difference maps (uint8 RGB) for human inspection:
        - diff_star: absolute delta from star reconstruction
        - diff_filament: absolute delta from filament repair
        - diff_dark_sky: absolute delta from dark sky stabilization
        - diff_accumulated: absolute delta between baseline and final fused output
        - diff_amplified: accumulated delta scaled by amplification_factor (default 15x)
        """
        base = baseline_rgb.astype(np.float32)
        fused = fused_rgb.astype(np.float32)

        # Accumulated diff
        accum_diff = np.abs(fused - base)
        diff_accum_img = np.clip(np.round(accum_diff), 0.0, 255.0).astype(np.uint8)

        # Amplified diff (15x gain for human visibility)
        amp_diff = np.clip(np.round(accum_diff * amplification_factor), 0.0, 255.0).astype(np.uint8)

        maps = {
            "accumulated": diff_accum_img,
            "amplified_15x": amp_diff
        }

        if star_reconstructed_rgb is not None:
            s_diff = np.abs(star_reconstructed_rgb.astype(np.float32) - base)
            maps["star_preserve"] = np.clip(np.round(s_diff * amplification_factor), 0.0, 255.0).astype(np.uint8)

        if filament_repaired_rgb is not None:
            f_diff = np.abs(filament_repaired_rgb.astype(np.float32) - base)
            maps["filament_guard"] = np.clip(np.round(f_diff * amplification_factor), 0.0, 255.0).astype(np.uint8)

        if dark_sky_rgb is not None:
            d_diff = np.abs(dark_sky_rgb.astype(np.float32) - base)
            maps["dark_sky"] = np.clip(np.round(d_diff * amplification_factor), 0.0, 255.0).astype(np.uint8)

        return maps

    def save_difference_maps(
        self,
        diff_maps: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "diff"
    ) -> Dict[str, str]:
        """Saves generated difference maps to disk."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        for name, arr in diff_maps.items():
            path = out_dir / f"{prefix}_{name}.png"
            Image.fromarray(arr).save(path)
            saved_paths[name] = str(path)

        return saved_paths

    def _compute_delta_stats(self, layer_delta: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Computes summary statistics for an individual layer's delta."""
        abs_d = np.abs(layer_delta)
        max_d = np.max(abs_d, axis=-1)
        active_pix = int(np.sum(mask > 0.02))
        eff_mod = int(np.sum(max_d >= 0.5))
        mean_abs = float(np.mean(abs_d))
        max_abs = float(np.max(abs_d))
        return {
            "candidate_pixels": active_pix,
            "modified_pixels": eff_mod,
            "mean_abs_diff": round(mean_abs, 4),
            "max_abs_diff": round(max_abs, 2)
        }
