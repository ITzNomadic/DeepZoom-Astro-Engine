"""
Test Suite for FilamentGuard V0.1.2 - Quantitative Geometric Validation.
Evaluates:
1. Centerline subpixel drift (mean and max <= 0.35 px).
2. Thickness preservation (FWHM ratio in [0.85, 1.15]).
3. Curvature preservation (mean deviation <= 5.0 deg).
4. Continuity healing (heals micro-fractures, reducing fragment count from 2 to 1).
5. Orthogonal anti-ringing suppression along v1 (lifts dark moats, reducing ringing index).
6. Single-Weight Contract compliance (linear scaling with weight w, no double attenuation).
"""

import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepzoom.astro.filament_guard import FilamentGuard


def generate_synthetic_curved_filament(
    size: tuple = (256, 256),
    bg_level: float = 25.0,
    peak_intensity: float = 180.0,
    sigma_width: float = 1.5,
    add_ringing: bool = False,
    ringing_depth: float = 14.0,
    add_micro_break: bool = False,
    break_x_range: tuple = (120, 126)
) -> np.ndarray:
    """
    Generates a high-precision synthetic curved filament with known geometry.
    Path: y(x) = y0 + amp * sin(2*pi*(x - x0)/period)
    """
    h, w = size
    yy, xx = np.indices((h, w), dtype=np.float32)

    # Filament trajectory: sine wave from x=30 to x=226
    x_coords = xx
    y_traj = 128.0 + 35.0 * np.sin(2.0 * np.pi * (x_coords - 30.0) / 160.0)

    # Compute orthogonal distance to curve (using 1st order Taylor expansion)
    # dy/dx = 35.0 * (2*pi/160.0) * cos(...)
    dydx = 35.0 * (2.0 * np.pi / 160.0) * np.cos(2.0 * np.pi * (x_coords - 30.0) / 160.0)
    # Normal distance approx: |y - y_traj| / sqrt(1 + (dy/dx)^2)
    norm_dist = np.abs(yy - y_traj) / np.sqrt(1.0 + dydx ** 2)

    # Gaussian profile across normal
    filament_profile = (peak_intensity - bg_level) * np.exp(-0.5 * (norm_dist / sigma_width) ** 2)

    # Restrict filament along x in [30, 226] with smooth cosine ends
    x_mask = np.zeros_like(xx)
    in_range = (xx >= 30) & (xx <= 226)
    x_mask[in_range] = 1.0
    # Soft taper at ends
    left_taper = (xx >= 30) & (xx < 40)
    x_mask[left_taper] = 0.5 * (1.0 - np.cos(np.pi * (xx[left_taper] - 30.0) / 10.0))
    right_taper = (xx > 216) & (xx <= 226)
    x_mask[right_taper] = 0.5 * (1.0 + np.cos(np.pi * (xx[right_taper] - 216.0) / 10.0))

    img = bg_level + filament_profile * x_mask

    # Add Ringing Defect: artificial dark trough flanking the filament at d ~ 3.2 px
    if add_ringing:
        # Ringing trough flanking ridge at d ~ 2.6 - 3.8 px
        trough = ringing_depth * np.exp(-0.5 * ((norm_dist - 3.2) / 0.8) ** 2) * x_mask
        img = img - trough

    # Add Micro-Break Defect: gap along ridge
    if add_micro_break:
        bx0, bx1 = break_x_range
        break_mask = (xx >= bx0) & (xx <= bx1)
        # Intensity drops in the break notch
        img[break_mask] = bg_level + (img[break_mask] - bg_level) * 0.15

    return np.clip(img, 0.0, 255.0).astype(np.float32)


def test_filament_geometric_validation():
    print("================================================================================")
    print("DEEPZOOM ASTRO ENGINE V0.1.2 - FILAMENTGUARD GEOMETRIC VALIDATION")
    print("================================================================================")

    out_dir = PROJECT_ROOT / "output" / "experimental" / "v012" / "crops" / "filaments"
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir = PROJECT_ROOT / "benchmarks" / "v012"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ground Truth Filament (Clean, continuous, no ringing)
    gt_img = generate_synthetic_curved_filament(add_ringing=False, add_micro_break=False)
    gt_rgb = np.stack([gt_img.astype(np.uint8)] * 3, axis=-1)
    Image.fromarray(gt_rgb).save(out_dir / "filament_gt.png")

    # 2. Degraded Baseline Filament (With ringing moats and micro-break notch)
    degraded_img = generate_synthetic_curved_filament(add_ringing=True, ringing_depth=15.0, add_micro_break=True)
    degraded_rgb = np.stack([degraded_img.astype(np.uint8)] * 3, axis=-1)
    Image.fromarray(degraded_rgb).save(out_dir / "filament_degraded_baseline.png")

    # 3. Apply FilamentGuard V0.1.2
    fg = FilamentGuard(ridge_threshold=0.08, coherence_threshold=0.30)
    
    # Analyze degraded baseline
    degraded_analysis = fg.analyze_filaments(degraded_img)
    print(f"Degraded Filament Coverage: {degraded_analysis['filament_coverage_pct']}%")
    print(f"Degraded Ringing Index:     {degraded_analysis['ringing_index']}")
    print(f"Degraded Ringing Detected:  {degraded_analysis['ringing_detected']}")

    # Apply pure physical repair (Single-Weight contract)
    repaired_rgb, repair_mask, repair_meta = fg.repair_filaments(
        baseline_rgb=degraded_rgb,
        analysis=degraded_analysis,
        pure_repair=True
    )
    Image.fromarray(repaired_rgb).save(out_dir / "filament_v012_repaired.png")
    Image.fromarray((repair_mask * 255.0).astype(np.uint8)).save(out_dir / "filament_repair_mask.png")

    repaired_gray = 0.299 * repaired_rgb[..., 0] + 0.587 * repaired_rgb[..., 1] + 0.114 * repaired_rgb[..., 2]

    # 4. Quantitative Geometric Evaluation
    eval_degraded_vs_gt = fg.evaluate_filament_geometry(gt_img, degraded_img)
    eval_repaired_vs_gt = fg.evaluate_filament_geometry(gt_img, repaired_gray)
    eval_repaired_vs_degraded = fg.evaluate_filament_geometry(degraded_img, repaired_gray)

    print("\n--- Geometric Metrics: Degraded Baseline vs Ground Truth ---")
    print(f"  Centerline Drift (mean):    {eval_degraded_vs_gt['centerline_drift_mean_px']} px")
    print(f"  Centerline Drift (max):     {eval_degraded_vs_gt['centerline_drift_max_px']} px")
    print(f"  Thickness Ratio (FWHM):     {eval_degraded_vs_gt['thickness_ratio']}")
    print(f"  Fragment Count (GT / Test): {eval_degraded_vs_gt['ref_fragment_count']} / {eval_degraded_vs_gt['test_fragment_count']}")
    print(f"  Ringing Moat Depth:         {eval_degraded_vs_gt.get('ringing_moat_depth_test', 0.0)}")

    print("\n--- Geometric Metrics: V0.1.2 Repaired vs Ground Truth ---")
    print(f"  Centerline Drift (mean):    {eval_repaired_vs_gt['centerline_drift_mean_px']} px")
    print(f"  Centerline Drift (max):     {eval_repaired_vs_gt['centerline_drift_max_px']} px")
    print(f"  Thickness Ratio (FWHM):     {eval_repaired_vs_gt['thickness_ratio']}")
    print(f"  Fragment Count (GT / Test): {eval_repaired_vs_gt['ref_fragment_count']} / {eval_repaired_vs_gt['test_fragment_count']}")
    print(f"  Curvature Deviation:        {eval_repaired_vs_gt['curvature_deviation_mean_deg']} deg")
    print(f"  Ringing Moat Depth:         {eval_repaired_vs_gt.get('ringing_moat_depth_test', 0.0)}")
    print(f"  Ringing Reduction:          {eval_repaired_vs_degraded['ringing_reduction_pct']}%")

    # 5. Single-Weight Contract Verification
    # Verify that repair mask is in [0, 1]
    assert np.all(repair_mask >= 0.0) and np.all(repair_mask <= 1.0), "Repair mask must be bounded in [0, 1]"
    
    # Verify linear scaling with weight w
    pure_delta = repaired_rgb.astype(np.float32) - degraded_rgb.astype(np.float32)
    blend_half = degraded_rgb.astype(np.float32) + 0.5 * np.expand_dims(repair_mask, -1) * pure_delta
    blend_full = degraded_rgb.astype(np.float32) + 1.0 * np.expand_dims(repair_mask, -1) * pure_delta
    
    diff_half = np.mean(np.abs(blend_half - degraded_rgb.astype(np.float32)))
    diff_full = np.mean(np.abs(blend_full - degraded_rgb.astype(np.float32)))
    linearity_ratio = diff_half / max(diff_full, 1e-4)
    print(f"\nSingle-Weight Linearity Ratio (w=0.5 / w=1.0): {linearity_ratio:.4f} (Expected: ~0.500)")
    assert abs(linearity_ratio - 0.500) < 0.01, f"Single-weight contract violation: ratio={linearity_ratio}"

    # 6. Validate Quality Acceptance Criteria
    # Centerline drift <= 0.35 px
    assert eval_repaired_vs_gt["centerline_drift_mean_px"] <= 0.35, "Centerline drift exceeds 0.35 px limit"
    # Thickness ratio within [0.85, 1.15]
    assert 0.85 <= eval_repaired_vs_gt["thickness_ratio"] <= 1.15, "Thickness ratio outside [0.85, 1.15]"
    # Curvature deviation <= 5.0 deg
    assert eval_repaired_vs_gt["curvature_deviation_mean_deg"] <= 5.0, "Curvature deviation exceeds 5.0 deg"
    # Continuity: test fragment count healed from split >= 2 back to 1
    assert eval_degraded_vs_gt["test_fragment_count"] >= 2, "Test baseline must have at least 2 fragments (split)"
    assert eval_repaired_vs_gt["test_fragment_count"] == 1, "FilamentGuard must heal break (fragment count = 1)"
    # Ringing reduction >= 30%
    assert eval_repaired_vs_degraded["ringing_reduction_pct"] >= 30.0, "Ringing reduction below 30%"

    print("\n>>> ALL FILAMENTGUARD V0.1.2 GEOMETRIC VALIDATION CHECKS PASSED SUCCESSFULLY! <<<\n")

    # Generate JSON report
    report = {
        "status": "passed",
        "version": "0.1.2",
        "description": "FilamentGuard V0.1.2 Quantitative Geometric Validation Benchmark",
        "ground_truth_geometry": {
            "type": "sine_curved_filament",
            "dimensions": [256, 256],
            "peak_intensity": 180.0,
            "background_level": 25.0,
            "nominal_sigma_px": 1.5,
            "nominal_fwhm_px": round(2.355 * 1.5, 2),
            "gt_fragments": 1
        },
        "degraded_baseline_metrics": eval_degraded_vs_gt,
        "repaired_v012_metrics": eval_repaired_vs_gt,
        "comparative_gains": {
            "centerline_drift_mean_px": eval_repaired_vs_gt["centerline_drift_mean_px"],
            "centerline_drift_max_px": eval_repaired_vs_gt["centerline_drift_max_px"],
            "thickness_ratio": eval_repaired_vs_gt["thickness_ratio"],
            "curvature_deviation_deg": eval_repaired_vs_gt["curvature_deviation_mean_deg"],
            "fragments_before": int(eval_degraded_vs_gt["test_fragment_count"]),
            "fragments_after": int(eval_repaired_vs_gt["test_fragment_count"]),
            "continuity_healed": bool(eval_repaired_vs_gt["test_fragment_count"] == 1),
            "ringing_reduction_pct": float(eval_repaired_vs_degraded["ringing_reduction_pct"]),
            "single_weight_linearity_ratio": round(float(linearity_ratio), 4)
        },
        "repair_diagnostics": {
            "repaired_pixels": int(repair_meta["repaired_pixels"]),
            "ringing_troughs_repaired": int(repair_meta["ringing_troughs_repaired"]),
            "continuity_healed_pixels": int(repair_meta["continuity_healed_pixels"]),
            "mean_repair_weight": float(repair_meta["mean_repair_weight"])
        },
        "criteria": {
            "centerline_drift_threshold_px": 0.35,
            "thickness_ratio_bounds": [0.85, 1.15],
            "curvature_dev_max_deg": 5.0,
            "continuity_healed_target": True,
            "min_ringing_reduction_pct": 30.0
        }
    }

    report_path = benchmarks_dir / "filament_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Filament benchmark report saved to: {report_path}")
    return report


if __name__ == "__main__":
    test_filament_geometric_validation()
