"""
DeepZoom Astro Engine V0.1 Benchmark Suite.
Validates StarPreserve, FilamentGuard, DarkSky, GeometryLock, and AstroValidator.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from deepzoom.pipeline import AstroPipeline
from deepzoom.utils.image_io import load_image
from deepzoom.validation import AstroValidator

def run_astro_engine_benchmark():
    pipeline = AstroPipeline()
    validator = AstroValidator()

    print("=" * 75)
    print("DEEPZOOM ASTRO ENGINE V0.1 - COMPREHENSIVE BENCHMARK")
    print("=" * 75)

    synthetic_lr = project_root / "samples" / "synthetic" / "synthetic_astro_input_512.png"
    synthetic_gt_path = project_root / "samples" / "synthetic" / "synthetic_astro_ground_truth_2048.png"
    catalog_path = project_root / "samples" / "synthetic" / "synthetic_ground_truth_catalog.json"

    with open(catalog_path, "r", encoding="utf-8") as f:
        gt_catalog = json.load(f)

    gt_arr = np.array(load_image(synthetic_gt_path))

    # BENCHMARK 1: Synthetic Dataset - Baseline Remacri vs Astro Engine V0.1
    out_base = project_root / "output" / "experimental" / "synthetic_baseline_remacri.png"
    out_astro = project_root / "output" / "experimental" / "synthetic_astro_engine.png"

    print("\n[BENCHMARK 1.A] Running Pure Frozen Remacri Baseline on Synthetic Dataset...")
    res_base = pipeline.run(
        input_image=synthetic_lr,
        output_image=out_base,
        scale=4,
        tile_size=256,
        enable_astro_engine=False
    )
    base_arr = np.array(load_image(out_base))
    psnr_base = validator.compute_psnr(gt_arr, base_arr)
    ssim_base = validator.compute_ssim(gt_arr, base_arr)

    print(f"  Remacri Baseline -> PSNR vs Ground Truth: {psnr_base} dB | SSIM: {ssim_base}")

    print("\n[BENCHMARK 1.B] Running DeepZoom Astro Engine V0.1 (StarPreserve + FilamentGuard + DarkSky)...")
    res_astro = pipeline.run(
        input_image=synthetic_lr,
        output_image=out_astro,
        scale=4,
        tile_size=256,
        enable_astro_engine=True,
        astro_blend_weight=0.5
    )
    astro_arr = np.array(load_image(out_astro))
    psnr_astro = validator.compute_psnr(gt_arr, astro_arr)
    ssim_astro = validator.compute_ssim(gt_arr, astro_arr)

    print(f"  Astro Engine V0.1 -> PSNR vs Ground Truth: {psnr_astro} dB | SSIM: {ssim_astro}")

    # StarPreserve Ground Truth Centroid Accuracy
    star_stats = res_astro["star_preserve"]
    print("\n[StarPreserve Metrics]")
    print(f"  Input Stars Detected:       {star_stats['input_stars_detected']}")
    print(f"  Output Stars Detected:      {star_stats['output_stars_detected']}")
    print(f"  Matched Stars:              {star_stats['matched_stars_count']}")
    print(f"  Spurious (Hallucinated):    {star_stats['spurious_stars_count']}")
    print(f"  Mean Centroid Drift:        {star_stats['mean_centroid_drift_px']} px")
    print(f"  Mean Flux Ratio:            {star_stats['mean_flux_preservation_ratio']}")

    # FilamentGuard Ringing and Coherence Metrics
    fg_stats = res_astro["filament_guard"]
    print("\n[FilamentGuard Metrics]")
    print(f"  Filament Coverage:          {fg_stats['filament_coverage_pct']}%")
    print(f"  Filament Coherence:         {fg_stats['mean_filament_coherence']}")
    print(f"  Ringing Index:              {fg_stats['ringing_index']}")
    print(f"  Ringing Detected:           {fg_stats['ringing_detected']}")

    # DarkSky Metrics
    ds_stats = res_astro["dark_sky"]
    print("\n[DarkSky Metrics]")
    print(f"  Background Noise Std:       {ds_stats['background_noise_std']}")
    print(f"  Pure Black Coverage:        {ds_stats['pure_black_coverage_pct']}%")
    print(f"  Clean Gradient Pass:        {ds_stats['clean_gradient_pass']}")

    # GeometryLock Metrics
    geo_stats = res_astro["geometry_lock"]
    print("\n[GeometryLock Metrics]")
    print(f"  Measured Shadow Scale:      {geo_stats['measured_scale']} (Expected: {geo_stats['expected_scale']})")
    print(f"  Scale Error:                {geo_stats['scale_error_pct']}%")
    print(f"  Centroid Drift:             {geo_stats['centroid_drift_px']} px")
    print(f"  Geometry Lock Status:       {'PASS' if geo_stats['geometric_lock_passed'] else 'FAIL'}")

    # Tiling Seam Continuity
    seams = res_astro["validation_metrics"]["tiling_seams"]
    print("\n[Tiling Seam Continuity]")
    print(f"  Mean Seam Discontinuity:    {seams['mean_seam_discontinuity']}")
    print(f"  Max Seam Discontinuity:     {seams['max_seam_discontinuity']}")
    print(f"  Seam Blending Passed:       {seams['seam_blending_passed']}")

    # BENCHMARK 2: Astronomical Black Hole Scene (4x to 8K Master)
    astro_input = project_root / "samples" / "astronomical" / "black_hole_base_1152x2048.png"
    astro_output = project_root / "output" / "experimental" / "blackhole_8k_astro_engine_v01.png"

    print("\n[BENCHMARK 2] Processing Real Black Hole Scene with Astro Engine V0.1...")
    t0 = time.time()
    res_bh = pipeline.run(
        input_image=astro_input,
        output_image=astro_output,
        scale=4,
        tile_size=400,
        enable_astro_engine=True,
        astro_blend_weight=0.5
    )
    t1 = time.time()
    print(f"  Completed 8K Black Hole Astro Master in {round(t1 - t0, 2)}s")
    print(f"  Dimensions: {res_bh['output_info']['dimensions']} ({res_bh['output_info']['megapixels']} MP)")

    # Save complete benchmark report
    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "AMD Radeon RX 5700 XT (GPU 0)",
        "synthetic_benchmark": {
            "psnr_remacri_baseline_db": psnr_base,
            "ssim_remacri_baseline": ssim_base,
            "psnr_astro_engine_v01_db": psnr_astro,
            "ssim_astro_engine_v01": ssim_astro,
            "star_preserve": star_stats,
            "filament_guard": fg_stats,
            "dark_sky": ds_stats,
            "geometry_lock": geo_stats,
            "tiling_seams": seams
        },
        "black_hole_scene": {
            "execution_time_seconds": round(t1 - t0, 2),
            "dimensions": res_bh["output_info"]["dimensions"],
            "megapixels": res_bh["output_info"]["megapixels"],
            "size_mb": res_bh["output_info"]["size_mb"],
            "star_preserve": res_bh["star_preserve"],
            "filament_guard": res_bh["filament_guard"],
            "dark_sky": res_bh["dark_sky"],
            "geometry_lock": res_bh["geometry_lock"]
        }
    }

    report_file = project_root / "benchmarks" / "astro_engine_v01_benchmark_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print(f"\n[SUCCESS] Comprehensive Benchmark Report written to: {report_file}")
    print("=" * 75)
    return benchmark_report

if __name__ == "__main__":
    run_astro_engine_benchmark()
