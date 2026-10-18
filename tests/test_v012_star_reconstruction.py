"""
Test Suite & Controlled Defect Injection Suite for StarPreserve V0.1.2.
Validates:
1. Subpixel recentering (0.5 px and 1.2 px drifts).
2. Explicit integrated flux conservation.
3. Negative ringing moat suppression.
4. Elongated and split core regularization.
5. Invariance / zero-regression on healthy control stars.
Outputs 1:1 comparative crops and benchmarks/v012/star_reconstruction_report.json.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from deepzoom.astro.star_preserve.star_analyzer import StarPreserve, StarRecord

def create_star_patch(
    size: int = 41,
    cx: float = 20.0,
    cy: float = 20.0,
    flux: float = 8000.0,
    sigma_x: float = 2.0,
    sigma_y: float = 2.0,
    bg: float = 25.0,
    ringing_amp: float = 0.0,
    ringing_rad: float = 5.0,
    split_second_peak: bool = False
) -> np.ndarray:
    yy, xx = np.indices((size, size))
    r_sq = ((xx - cx) ** 2) / (2.0 * sigma_x ** 2) + ((yy - cy) ** 2) / (2.0 * sigma_y ** 2)
    norm = 2.0 * np.pi * sigma_x * sigma_y
    star = bg + (flux / norm) * np.exp(-r_sq)

    if split_second_peak:
        r_sq2 = ((xx - (cx + 2.5)) ** 2) / (2.0 * 1.5 ** 2) + ((yy - cy) ** 2) / (2.0 * 1.5 ** 2)
        star += (0.4 * flux / (2.0 * np.pi * 1.5**2)) * np.exp(-r_sq2)

    if ringing_amp > 0:
        r_dist = np.hypot(xx - cx, yy - cy)
        moat = ringing_amp * np.exp(-((r_dist - ringing_rad) ** 2) / (2.0 * 1.2 ** 2))
        star -= moat

    return np.clip(star, 0.0, 255.0).astype(np.float32)


def run_star_reconstruction_suite():
    print("=" * 75)
    print("STARPRESERVE V0.1.2 - CONTROLLED DEFECT INJECTION SUITE")
    print("=" * 75)

    sp = StarPreserve()
    crops_dir = project_root / "output" / "experimental" / "v012" / "crops" / "star_defects"
    crops_dir.mkdir(parents=True, exist_ok=True)

    size = 41
    center = 20.0
    bg_level = 25.0
    expected_flux = 8000.0

    # Ground truth reference star
    gt_patch = create_star_patch(size=size, cx=center, cy=center, flux=expected_flux, sigma_x=2.0, sigma_y=2.0, bg=bg_level)

    defects = [
        {
            "name": "Defect_A_Small_Drift_0.5px",
            "desc": "Subpixel drift of 0.5 pixels in X",
            "patch": create_star_patch(size=size, cx=center + 0.5, cy=center, flux=expected_flux, sigma_x=2.0, sigma_y=2.0, bg=bg_level),
            "true_cx": center,
            "true_cy": center
        },
        {
            "name": "Defect_B_Moderate_Drift_1.2px",
            "desc": "Subpixel drift of 1.2 pixels diagonally (+0.8 px X, -0.9 px Y)",
            "patch": create_star_patch(size=size, cx=center + 0.8, cy=center - 0.9, flux=expected_flux, sigma_x=2.0, sigma_y=2.0, bg=bg_level),
            "true_cx": center,
            "true_cy": center
        },
        {
            "name": "Defect_C_Negative_Ringing_Moat",
            "desc": "Artificial Gibbs ringing moat dipping 22 levels below local background",
            "patch": create_star_patch(size=size, cx=center, cy=center, flux=expected_flux, sigma_x=2.0, sigma_y=2.0, bg=bg_level, ringing_amp=22.0, ringing_rad=5.5),
            "true_cx": center,
            "true_cy": center
        },
        {
            "name": "Defect_D_Split_Duplicated_Core",
            "desc": "Asymmetric core splitting / artificial elongation",
            "patch": create_star_patch(size=size, cx=center - 1.0, cy=center, flux=expected_flux * 0.7, sigma_x=1.8, sigma_y=1.8, bg=bg_level, split_second_peak=True),
            "true_cx": center,
            "true_cy": center
        },
        {
            "name": "Defect_E_Elongated_Deformation",
            "desc": "Neural elongation along one axis (ellipticity = 0.60)",
            "patch": create_star_patch(size=size, cx=center, cy=center, flux=expected_flux, sigma_x=3.2, sigma_y=1.4, bg=bg_level),
            "true_cx": center,
            "true_cy": center
        },
        {
            "name": "Defect_F_Flux_Deficit_30pct",
            "desc": "30% flux deficit caused by over-sharpening",
            "patch": create_star_patch(size=size, cx=center, cy=center, flux=expected_flux * 0.70, sigma_x=1.8, sigma_y=1.8, bg=bg_level),
            "true_cx": center,
            "true_cy": center
        },
        {
            "name": "Control_G_Healthy_Star",
            "desc": "Healthy point source without defects (tests zero regression)",
            "patch": gt_patch.copy(),
            "true_cx": center,
            "true_cy": center
        }
    ]

    results = []

    # Mock matching input record (representing projected LR source)
    s_in = StarRecord(
        star_id=0,
        peak_x=int(center),
        peak_y=int(center),
        subpixel_x=center,
        subpixel_y=center,
        flux=expected_flux,  # in HR units (already scaled by s^2)
        fwhm=4.71,
        ellipticity=0.0,
        source_type="point_source"
    )

    for d in defects:
        name = d["name"]
        print(f"\n[TESTING] {name}")
        print(f"  Description: {d['desc']}")
        deg_patch = d["patch"]

        # 3-channel RGB representation
        deg_rgb = np.stack([deg_patch, deg_patch, deg_patch], axis=-1).astype(np.uint8)
        gt_rgb = np.stack([gt_patch, gt_patch, gt_patch], axis=-1).astype(np.uint8)

        # Measure degraded star properties
        deg_gray = deg_patch
        net_deg = np.maximum(deg_gray - bg_level, 0.0)
        f_deg = float(np.sum(net_deg))
        yy, xx = np.indices((size, size))
        cx_deg = float(np.sum(xx * net_deg) / max(f_deg, 1e-3))
        cy_deg = float(np.sum(yy * net_deg) / max(f_deg, 1e-3))
        drift_before = float(np.hypot(cx_deg - center, cy_deg - center))
        min_before = float(np.min(deg_gray))
        flux_err_before = float(abs(f_deg - expected_flux) / expected_flux * 100.0)

        # Mock output detection
        s_out = StarRecord(
            star_id=0,
            peak_x=int(round(cx_deg)),
            peak_y=int(round(cy_deg)),
            subpixel_x=cx_deg,
            subpixel_y=cy_deg,
            flux=f_deg,
            fwhm=4.71,
            ellipticity=0.0,
            source_type="point_source",
            confidence=1.0
        )

        # Apply StarPreserve V0.1.2 Reconstruction
        reconstructed_rgb, conf_mask, meta = sp.reconstruct_point_sources(
            baseline_rgb=deg_rgb,
            input_stars=[s_in],
            output_stars=[s_out],
            scale_factor=1.0  # 1.0 because patch is already in high-res coordinates
        )

        rec_gray = reconstructed_rgb[..., 0].astype(np.float32)
        net_rec = np.maximum(rec_gray - bg_level, 0.0)
        f_rec = float(np.sum(net_rec))
        cx_rec = float(np.sum(xx * net_rec) / max(f_rec, 1e-3))
        cy_rec = float(np.sum(yy * net_rec) / max(f_rec, 1e-3))
        drift_after = float(np.hypot(cx_rec - center, cy_rec - center))
        min_after = float(np.min(rec_gray))
        flux_err_after = float(abs(f_rec - expected_flux) / expected_flux * 100.0)

        # PSNR vs Ground Truth (using uint8 reference consistent with reconstructed image format)
        gt_ref = gt_rgb[..., 0].astype(np.float32)
        deg_eval = deg_rgb[..., 0].astype(np.float32)
        rec_eval = reconstructed_rgb[..., 0].astype(np.float32)

        mse_before = float(np.mean((deg_eval - gt_ref)**2))
        psnr_before = 10.0 * np.log10(255.0**2 / max(mse_before, 1e-10))
        mse_after = float(np.mean((rec_eval - gt_ref)**2))
        psnr_after = 10.0 * np.log10(255.0**2 / max(mse_after, 1e-10))

        print(f"  Centroid Drift:  Before: {drift_before:.3f} px  ->  After: {drift_after:.3f} px")
        print(f"  Flux Error:      Before: {flux_err_before:.1f}%    ->  After: {flux_err_after:.1f}%")
        print(f"  Min Intensity:   Before: {min_before:.1f}      ->  After: {min_after:.1f} (Sky: {bg_level:.1f})")
        print(f"  PSNR vs GT:      Before: {psnr_before:.2f} dB ->  After: {psnr_after:.2f} dB (Gain: {psnr_after - psnr_before:+.2f} dB)")

        # Save 1:1 comparative crops (scaled 8x with Nearest Neighbor for easy visual inspection)
        scale_vis = 8
        im_gt = Image.fromarray(gt_rgb).resize((size * scale_vis, size * scale_vis), Image.Resampling.NEAREST)
        im_deg = Image.fromarray(deg_rgb).resize((size * scale_vis, size * scale_vis), Image.Resampling.NEAREST)
        im_rec = Image.fromarray(reconstructed_rgb).resize((size * scale_vis, size * scale_vis), Image.Resampling.NEAREST)
        
        # Amplified difference (x10)
        diff_arr = np.abs(reconstructed_rgb.astype(np.float32) - deg_rgb.astype(np.float32))
        diff_amplified = np.clip(diff_arr * 10.0, 0.0, 255.0).astype(np.uint8)
        im_diff = Image.fromarray(diff_amplified).resize((size * scale_vis, size * scale_vis), Image.Resampling.NEAREST)

        im_gt.save(crops_dir / f"{name}_01_ground_truth.png")
        im_deg.save(crops_dir / f"{name}_02_degraded.png")
        im_rec.save(crops_dir / f"{name}_03_starpreserve_corrected.png")
        im_diff.save(crops_dir / f"{name}_04_diff_amplified_x10.png")

        results.append({
            "defect_name": name,
            "description": d["desc"],
            "metrics": {
                "centroid_drift_before_px": round(drift_before, 3),
                "centroid_drift_after_px": round(drift_after, 3),
                "flux_error_before_pct": round(flux_err_before, 2),
                "flux_error_after_pct": round(flux_err_after, 2),
                "min_intensity_before": round(min_before, 2),
                "min_intensity_after": round(min_after, 2),
                "psnr_before_db": round(psnr_before, 2),
                "psnr_after_db": round(psnr_after, 2),
                "psnr_gain_db": round(psnr_after - psnr_before, 2)
            },
            "status": "IMPROVED" if psnr_after > psnr_before + 0.05 else ("PRESERVED" if abs(psnr_after - psnr_before) <= 0.05 else "REGRESSED")
        })

    report = {
        "suite": "StarPreserve_V0.1.2_Controlled_Defect_Injection",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_flux": expected_flux,
        "sky_background": bg_level,
        "defects_evaluated": len(results),
        "crops_saved_directory": str(crops_dir),
        "results": results
    }

    report_dir = project_root / "benchmarks" / "v012"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "star_reconstruction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Star Reconstruction Report written to: {report_path}")
    print(f"[OK] 1:1 Comparative crops saved to: {crops_dir}")
    print("=" * 75)
    return report

if __name__ == "__main__":
    run_star_reconstruction_suite()
