"""
Test Suite for AdaptiveFusion V0.1.2 - Single-Weight Contract & Metric Synchronization.
Evaluates:
1. Single-Weight Contract: zero double attenuation, strict linear scaling with weight w.
2. Bitwise preservation at w=0.0 or mode='baseline' (100% bitwise identical).
3. Synchronization between AdaptiveFuser and AstroValidator spatial modification metrics.
4. Difference maps generation and export (individual layers, accumulated, and 15x amplified).
5. Percentile delta tracking (p50, p90, p99, mean, max).
"""

import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepzoom.fusion.adaptive_fuser import AdaptiveFuser
from deepzoom.validation.astro_validator import AstroValidator


def test_adaptive_fusion_v012():
    print("================================================================================")
    print("DEEPZOOM ASTRO ENGINE V0.1.2 - ADAPTIVE FUSION VALIDATION")
    print("================================================================================")

    out_dir = PROJECT_ROOT / "output" / "experimental" / "v012" / "diff_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmarks_dir = PROJECT_ROOT / "benchmarks" / "v012"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    h, w_img = 256, 256
    np.random.seed(42)

    # 1. Synthetic baseline
    baseline = np.full((h, w_img, 3), 20, dtype=np.uint8)

    # 2. Synthetic Star correction (pure delta +40 on star patch)
    star_reconstructed = baseline.copy()
    star_reconstructed[40:60, 40:60, :] = 160
    star_mask = np.zeros((h, w_img), dtype=np.float32)
    star_mask[40:60, 40:60] = 0.95

    # 3. Synthetic Filament correction (pure delta +25 along ridge)
    filament_repaired = baseline.copy()
    filament_repaired[120:130, 20:200, :] = 95
    filament_mask = np.zeros((h, w_img), dtype=np.float32)
    filament_mask[120:130, 20:200] = 0.85

    # 4. Synthetic DarkSky stabilization (pure delta -3 on flat background)
    dark_sky = baseline.copy().astype(np.int32)
    dark_sky[180:240, 180:240, :] = 17
    dark_sky_rgb = np.clip(dark_sky, 0, 255).astype(np.uint8)
    dark_mask = np.zeros((h, w_img), dtype=np.float32)
    dark_mask[180:240, 180:240] = 0.70

    fuser = AdaptiveFuser(default_weight=0.5)
    validator = AstroValidator()

    # ---------------------------------------------------------
    # Test A: Bitwise preservation at w=0.0 or mode='baseline'
    # ---------------------------------------------------------
    fused_base, stats_base = fuser.fuse(
        baseline_rgb=baseline,
        mode="baseline",
        star_reconstructed_rgb=star_reconstructed,
        star_mask=star_mask,
        filament_repaired_rgb=filament_repaired,
        filament_mask=filament_mask,
        dark_sky_rgb=dark_sky_rgb,
        dark_sky_mask=dark_mask,
        weight=0.5
    )
    assert np.array_equal(fused_base, baseline), "Baseline mode must be 100% bitwise identical"
    assert stats_base["effectively_modified_pixels"] == 0, "Modified pixels must be 0 for baseline"

    fused_w0, stats_w0 = fuser.fuse(
        baseline_rgb=baseline,
        mode="all_layers",
        star_reconstructed_rgb=star_reconstructed,
        star_mask=star_mask,
        filament_repaired_rgb=filament_repaired,
        filament_mask=filament_mask,
        dark_sky_rgb=dark_sky_rgb,
        dark_sky_mask=dark_mask,
        weight=0.0
    )
    assert np.array_equal(fused_w0, baseline), "Weight 0.0 must be 100% bitwise identical"

    # ---------------------------------------------------------
    # Test B: Strict Single-Weight Contract Linearity
    # ---------------------------------------------------------
    weights = [0.25, 0.50, 0.75, 1.00]
    deltas = []
    for w in weights:
        fused_w, stats_w = fuser.fuse(
            baseline_rgb=baseline,
            mode="all_layers",
            star_reconstructed_rgb=star_reconstructed,
            star_mask=star_mask,
            filament_repaired_rgb=filament_repaired,
            filament_mask=filament_mask,
            dark_sky_rgb=dark_sky_rgb,
            dark_sky_mask=dark_mask,
            weight=w
        )
        mean_delta = stats_w["mean_absolute_difference"]
        deltas.append(mean_delta)
        print(f"  Weight w={w:.2f} -> Mean Absolute Delta = {mean_delta:.4f}")

    # Verify linear proportionality
    ratio_50_to_100 = deltas[1] / deltas[3]
    ratio_25_to_100 = deltas[0] / deltas[3]
    ratio_75_to_100 = deltas[2] / deltas[3]
    print(f"\nLinearity Check (w=0.50/1.00): {ratio_50_to_100:.4f} (Expected: ~0.500)")
    print(f"Linearity Check (w=0.25/1.00): {ratio_25_to_100:.4f} (Expected: ~0.250)")
    print(f"Linearity Check (w=0.75/1.00): {ratio_75_to_100:.4f} (Expected: ~0.750)")

    assert abs(ratio_50_to_100 - 0.50) < 0.015, "Linearity ratio 0.50 violation"
    assert abs(ratio_25_to_100 - 0.25) < 0.015, "Linearity ratio 0.25 violation"
    assert abs(ratio_75_to_100 - 0.75) < 0.015, "Linearity ratio 0.75 violation"

    # ---------------------------------------------------------
    # Test C: Synchronization with AstroValidator
    # ---------------------------------------------------------
    fused_final, stats_final = fuser.fuse(
        baseline_rgb=baseline,
        mode="all_layers",
        star_reconstructed_rgb=star_reconstructed,
        star_mask=star_mask,
        filament_repaired_rgb=filament_repaired,
        filament_mask=filament_mask,
        dark_sky_rgb=dark_sky_rgb,
        dark_sky_mask=dark_mask,
        weight=0.5
    )
    val_report = validator.evaluate_all(baseline, fused_final)
    val_delta = val_report["spatial_delta"]

    print("\n--- Synchronization: AdaptiveFuser vs AstroValidator ---")
    print(f"  Fuser Effectively Modified Pixels: {stats_final['effectively_modified_pixels']}")
    print(f"  Validator Modified Pixels:        {val_delta['modified_pixel_count']}")
    print(f"  Fuser Modified %:                 {stats_final['effectively_modified_pct']}%")
    print(f"  Validator Modified %:             {val_delta['modified_pixel_pct']}%")
    print(f"  Fuser Mean Abs Diff:              {stats_final['mean_absolute_difference']}")
    print(f"  Validator Mean Abs Diff:          {val_delta['mean_absolute_difference']}")
    print(f"  Fuser Max Abs Diff:               {stats_final['max_absolute_difference']}")
    print(f"  Validator Max Abs Diff:           {val_delta['max_absolute_difference']}")
    print(f"  Fuser p50 / p90 / p99:            {stats_final['p50_difference']} / {stats_final['p90_difference']} / {stats_final['p99_difference']}")
    print(f"  Validator p50 / p90 / p99:        {val_delta['p50_difference']} / {val_delta['p90_difference']} / {val_delta['p99_difference']}")

    # Assert exact synchronization
    assert stats_final["effectively_modified_pixels"] == val_delta["modified_pixel_count"], "Modified count desynchronized!"
    assert stats_final["effectively_modified_pct"] == val_delta["modified_pixel_pct"], "Modified pct desynchronized!"
    assert stats_final["mean_absolute_difference"] == val_delta["mean_absolute_difference"], "Mean diff desynchronized!"
    assert stats_final["max_absolute_difference"] == val_delta["max_absolute_difference"], "Max diff desynchronized!"
    assert stats_final["p50_difference"] == val_delta["p50_difference"], "p50 desynchronized!"
    assert stats_final["p90_difference"] == val_delta["p90_difference"], "p90 desynchronized!"
    assert stats_final["p99_difference"] == val_delta["p99_difference"], "p99 desynchronized!"

    # ---------------------------------------------------------
    # Test D: Difference Maps Generation and Export
    # ---------------------------------------------------------
    diff_maps = fuser.generate_difference_maps(
        baseline_rgb=baseline,
        fused_rgb=fused_final,
        star_reconstructed_rgb=star_reconstructed,
        filament_repaired_rgb=filament_repaired,
        dark_sky_rgb=dark_sky_rgb,
        amplification_factor=15.0
    )
    saved_paths = fuser.save_difference_maps(diff_maps, out_dir, prefix="test_fusion")
    print("\n--- Difference Maps Exported ---")
    for name, p in saved_paths.items():
        print(f"  {name}: {p}")
        assert Path(p).exists(), f"Diff map {p} does not exist"

    # Verify amplified map has expected 15x gain
    amp_map = diff_maps["amplified_15x"]
    acc_map = diff_maps["accumulated"]
    max_acc = int(np.max(acc_map))
    max_amp = int(np.max(amp_map))
    print(f"  Max Accumulated Delta: {max_acc}, Max Amplified (15x): {max_amp}")
    assert max_amp >= min(255, max_acc * 14), "Amplification gain not applied"

    print("\n>>> ALL ADAPTIVEFUSER V0.1.2 VALIDATION CHECKS PASSED SUCCESSFULLY! <<<\n")

    report = {
        "status": "passed",
        "version": "0.1.2",
        "description": "AdaptiveFusion V0.1.2 Single-Weight Contract and Metric Synchronization",
        "single_weight_contract": {
            "linearity_ratios": {
                "w0.50_to_1.00": round(ratio_50_to_100, 4),
                "w0.25_to_1.00": round(ratio_25_to_100, 4),
                "w0.75_to_1.00": round(ratio_75_to_100, 4)
            },
            "status": "VERIFIED_LINEAR"
        },
        "metric_synchronization": {
            "effectively_modified_pixels": stats_final["effectively_modified_pixels"],
            "effectively_modified_pct": stats_final["effectively_modified_pct"],
            "mean_absolute_difference": stats_final["mean_absolute_difference"],
            "max_absolute_difference": stats_final["max_absolute_difference"],
            "p50_difference": stats_final["p50_difference"],
            "p90_difference": stats_final["p90_difference"],
            "p99_difference": stats_final["p99_difference"],
            "synchronization_status": "EXACT_MATCH"
        },
        "difference_maps": saved_paths
    }

    report_path = benchmarks_dir / "fusion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Fusion benchmark report saved to: {report_path}")
    return report


if __name__ == "__main__":
    test_adaptive_fusion_v012()
