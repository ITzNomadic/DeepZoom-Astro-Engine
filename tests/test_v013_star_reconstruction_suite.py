"""
StarPreserve V0.1.3 - Comprehensive Controlled Defect & Photometric Lock Suite.
Tests:
1. Subpixel drift (0.5 px)
2. Subpixel drift (1.2 px)
3. Excessive halo / negative ringing moat
4. Split / duplicated core
5. Elongated star (astigmatism)
6. Partially saturated star
7. Nearby crowded stars (binary separation)
8. Non-uniform background (emission nebulosity)
9. Healthy control star (zero regression guarantee)

Generates: benchmarks/v013/star_reconstruction_report.json
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepzoom.astro.star_preserve import StarPreserve
from deepzoom.astro.star_preserve.star_analyzer import StarRecord


def generate_synthetic_star_case(
    name: str,
    size: int = 41,
    center: float = 20.0,
    bg_level: float = 25.0,
    peak_amp: float = 180.0,
    sigma_x: float = 2.2,
    sigma_y: float = 2.2,
    defect_type: str = "none"
):
    yy, xx = np.indices((size, size), dtype=np.float32)
    
    # Ground truth clean Gaussian star
    gt_patch = bg_level + peak_amp * np.exp(-0.5 * (((xx - center)/sigma_x)**2 + ((yy - center)/sigma_y)**2))
    expected_flux = float(2.0 * np.pi * peak_amp * sigma_x * sigma_y)
    
    deg_patch = gt_patch.copy()
    desc = ""

    if defect_type == "drift_05":
        desc = "Subpixel centroid drift of 0.5 px in X"
        deg_patch = bg_level + peak_amp * np.exp(-0.5 * (((xx - (center + 0.5))/sigma_x)**2 + ((yy - center)/sigma_y)**2))
    elif defect_type == "drift_12":
        desc = "Subpixel centroid drift of 1.2 px diagonally (+0.8 px X, -0.9 px Y)"
        deg_patch = bg_level + peak_amp * np.exp(-0.5 * (((xx - (center + 0.8))/sigma_x)**2 + ((yy - (center - 0.9))/sigma_y)**2))
    elif defect_type == "ringing_halo":
        desc = "Deep Gibbs ringing moat dipping 20 levels below local background"
        r = np.hypot(xx - center, yy - center)
        moat = 20.0 * np.exp(-0.5 * ((r - 5.5) / 1.2)**2)
        deg_patch = np.maximum(gt_patch - moat, 5.0)
    elif defect_type == "split_core":
        desc = "Asymmetric core splitting into dual peaks"
        p1 = 0.55 * peak_amp * np.exp(-0.5 * (((xx - (center - 1.2))/1.6)**2 + ((yy - center)/1.6)**2))
        p2 = 0.55 * peak_amp * np.exp(-0.5 * (((xx - (center + 1.2))/1.6)**2 + ((yy - center)/1.6)**2))
        deg_patch = bg_level + p1 + p2
    elif defect_type == "elongated":
        desc = "Artificial neural elongation (ellipticity e=0.60)"
        deg_patch = bg_level + peak_amp * np.exp(-0.5 * (((xx - center)/4.0)**2 + ((yy - center)/1.6)**2))
    elif defect_type == "saturated":
        desc = "Saturated flat-topped stellar core (clipped at 240)"
        deg_patch = np.clip(gt_patch * 1.35, 0.0, 240.0)
    elif defect_type == "nearby_crowded":
        desc = "Close binary pair separated by 4.5 px"
        sec_center_x = center + 4.5
        sec_center_y = center - 2.0
        sec_star = 0.65 * peak_amp * np.exp(-0.5 * (((xx - sec_center_x)/1.8)**2 + ((yy - sec_center_y)/1.8)**2))
        gt_patch = gt_patch + sec_star
        deg_patch = gt_patch.copy()
    elif defect_type == "non_uniform_nebulosity":
        desc = "Non-uniform emission nebula gradient across aperture"
        nebula_grad = 20.0 + 15.0 * np.sin(xx / 8.0) + 10.0 * np.cos(yy / 10.0)
        gt_patch = gt_patch + nebula_grad
        deg_patch = gt_patch.copy()
    elif defect_type == "healthy":
        desc = "Pristine healthy star for zero-regression evaluation"
        deg_patch = gt_patch.copy()

    return {
        "name": name,
        "desc": desc,
        "gt_patch": gt_patch,
        "deg_patch": deg_patch,
        "center_x": center,
        "center_y": center,
        "expected_flux": expected_flux,
        "bg_level": bg_level
    }


def run_star_reconstruction_suite():
    print("================================================================================")
    print("DEEPZOOM ASTRO ENGINE V0.1.3 - STARPRESERVE PHOTOMETRIC LOCK SUITE")
    print("================================================================================")

    test_cases = [
        generate_synthetic_star_case("case1_subpixel_drift_05px", defect_type="drift_05"),
        generate_synthetic_star_case("case2_subpixel_drift_12px", defect_type="drift_12"),
        generate_synthetic_star_case("case3_excessive_ringing_halo", defect_type="ringing_halo"),
        generate_synthetic_star_case("case4_split_duplicated_core", defect_type="split_core"),
        generate_synthetic_star_case("case5_elongated_star", defect_type="elongated"),
        generate_synthetic_star_case("case6_partially_saturated_star", defect_type="saturated"),
        generate_synthetic_star_case("case7_nearby_crowded_stars", defect_type="nearby_crowded"),
        generate_synthetic_star_case("case8_non_uniform_nebulosity", defect_type="non_uniform_nebulosity"),
        generate_synthetic_star_case("case9_healthy_control_star", defect_type="healthy")
    ]

    sp = StarPreserve()
    results = []

    out_crops_dir = PROJECT_ROOT / "output" / "experimental" / "v013" / "crops" / "star_photometric_lock"
    out_crops_dir.mkdir(parents=True, exist_ok=True)

    for case in test_cases:
        cname = case["name"]
        cdesc = case["desc"]
        gt_arr = case["gt_patch"]
        deg_arr = case["deg_patch"]
        cx = case["center_x"]
        cy = case["center_y"]
        bg = case["bg_level"]
        exp_flux = case["expected_flux"]
        h, w = deg_arr.shape

        gt_rgb = np.stack([np.clip(np.round(gt_arr), 0, 255).astype(np.uint8)] * 3, axis=-1)
        deg_rgb = np.stack([np.clip(np.round(deg_arr), 0, 255).astype(np.uint8)] * 3, axis=-1)

        # Baseline metrics
        net_deg = np.maximum(deg_rgb[..., 0].astype(np.float32) - bg, 0.0)
        f_deg = float(np.sum(net_deg))
        yy, xx = np.indices((h, w))
        meas_cx_deg = float(np.sum(xx * net_deg) / max(f_deg, 1e-3))
        meas_cy_deg = float(np.sum(yy * net_deg) / max(f_deg, 1e-3))
        drift_before = float(np.hypot(meas_cx_deg - cx, meas_cy_deg - cy))
        flux_err_before = float(abs(f_deg - exp_flux) / exp_flux * 100.0)
        min_int_before = float(np.min(deg_rgb[..., 0]))

        # Reference input star record
        s_in = StarRecord(
            star_id=1,
            peak_x=int(cx),
            peak_y=int(cy),
            subpixel_x=cx,
            subpixel_y=cy,
            flux=exp_flux,
            fwhm=4.8,
            ellipticity=0.0,
            source_type="point_source",
            confidence=1.0
        )

        s_out = StarRecord(
            star_id=1,
            peak_x=int(round(meas_cx_deg)),
            peak_y=int(round(meas_cy_deg)),
            subpixel_x=meas_cx_deg,
            subpixel_y=meas_cy_deg,
            flux=f_deg,
            fwhm=4.8,
            ellipticity=0.0,
            source_type="point_source",
            confidence=1.0
        )

        # Apply V0.1.3 Reconstructor with Photometric Lock
        rec_rgb, conf_mask, meta = sp.reconstruct_point_sources(
            baseline_rgb=deg_rgb,
            input_stars=[s_in],
            output_stars=[s_out],
            scale_factor=1.0,
            pure_repair=True
        )

        # Reconstructed metrics
        net_rec = np.maximum(rec_rgb[..., 0].astype(np.float32) - bg, 0.0)
        f_rec = float(np.sum(net_rec))
        meas_cx_rec = float(np.sum(xx * net_rec) / max(f_rec, 1e-3))
        meas_cy_rec = float(np.sum(yy * net_rec) / max(f_rec, 1e-3))
        drift_after = float(np.hypot(meas_cx_rec - cx, meas_cy_rec - cy))
        flux_err_after = float(abs(f_rec - exp_flux) / exp_flux * 100.0)
        min_int_after = float(np.min(rec_rgb[..., 0]))

        # PSNR vs uint8 Ground Truth
        gt_ref = gt_rgb[..., 0].astype(np.float32)
        deg_ref = deg_rgb[..., 0].astype(np.float32)
        rec_ref = rec_rgb[..., 0].astype(np.float32)

        mse_before = float(np.mean((deg_ref - gt_ref) ** 2))
        psnr_before = 10.0 * np.log10(255.0 ** 2 / max(mse_before, 1e-10))
        mse_after = float(np.mean((rec_ref - gt_ref) ** 2))
        psnr_after = 10.0 * np.log10(255.0 ** 2 / max(mse_after, 1e-10))
        psnr_gain = psnr_after - psnr_before

        print(f"\n[{cname.upper()}] — {cdesc}")
        print(f"  Centroid Drift:  Before: {drift_before:6.3f} px  ->  After: {drift_after:6.3f} px")
        print(f"  Flux Error:      Before: {flux_err_before:6.2f}%   ->  After: {flux_err_after:6.2f}%")
        print(f"  Min Intensity:   Before: {min_int_before:6.1f}     ->  After: {min_int_after:6.1f} (Sky: {bg:.1f})")
        print(f"  PSNR vs GT:      Before: {psnr_before:6.2f} dB ->  After: {psnr_after:6.2f} dB (Gain: {psnr_gain:+6.2f} dB)")

        # Save comparative crops (scaled 8x nearest neighbor)
        scale_vis = 8
        Image.fromarray(gt_rgb).resize((w * scale_vis, h * scale_vis), Image.Resampling.NEAREST).save(out_crops_dir / f"{cname}_01_gt.png")
        Image.fromarray(deg_rgb).resize((w * scale_vis, h * scale_vis), Image.Resampling.NEAREST).save(out_crops_dir / f"{cname}_02_degraded.png")
        Image.fromarray(rec_rgb).resize((w * scale_vis, h * scale_vis), Image.Resampling.NEAREST).save(out_crops_dir / f"{cname}_03_reconstructed.png")

        diff = np.abs(rec_ref - deg_ref)
        diff_vis = np.clip(diff * 15.0, 0, 255).astype(np.uint8)
        diff_3d = np.stack([diff_vis]*3, -1)
        Image.fromarray(diff_3d).resize((w * scale_vis, h * scale_vis), Image.Resampling.NEAREST).save(out_crops_dir / f"{cname}_04_diff_15x.png")

        status = "IMPROVED" if psnr_gain > 0.05 else ("PRESERVED" if abs(psnr_gain) <= 0.05 else "REGRESSED")
        results.append({
            "test_case": cname,
            "description": cdesc,
            "status": status,
            "metrics": {
                "centroid_drift_before_px": round(drift_before, 3),
                "centroid_drift_after_px": round(drift_after, 3),
                "flux_error_before_pct": round(flux_err_before, 2),
                "flux_error_after_pct": round(flux_err_after, 2),
                "flux_delta_pct": round(flux_err_after - flux_err_before, 2),
                "min_intensity_before": round(min_int_before, 1),
                "min_intensity_after": round(min_int_after, 1),
                "psnr_before_db": round(psnr_before, 2),
                "psnr_after_db": round(psnr_after, 2),
                "psnr_gain_db": round(psnr_gain, 2)
            }
        })

    out_json = {
        "version": "0.1.3",
        "benchmark_name": "StarPreserve_Photometric_Lock_Controlled_Suite",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases_evaluated": len(results),
        "results": results,
        "fidelity_gate_summary": sp.fidelity_gate.get_audit_summary()
    }

    out_path = PROJECT_ROOT / "benchmarks" / "v013" / "star_reconstruction_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    run_star_reconstruction_suite()
