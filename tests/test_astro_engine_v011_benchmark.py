"""
DeepZoom Astro Engine V0.1.1 Scientific Benchmark & Ablation Study Suite.
Evaluates 6 ablation modes on the expanded synthetic dataset (Groups A, B, C, D) against Ground Truth
and processes the real astronomical reference scene (deepzoom_astronomy_test.png).
Outputs benchmarks/astro_engine_v011_benchmark_report.json.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from deepzoom.pipeline import AstroPipeline
from deepzoom.utils.image_io import load_image, get_image_info
from deepzoom.validation import AstroValidator

def run_v011_benchmark():
    pipeline = AstroPipeline()
    validator = AstroValidator()

    print("=" * 80)
    print("DEEPZOOM ASTRO ENGINE V0.1.1 - RIGOROUS SCIENTIFIC BENCHMARK & ABLATION STUDY")
    print("=" * 80)

    exp_dir = project_root / "output" / "experimental" / "v011"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # PART 1: EXPANDED SYNTHETIC SUITE (4x scale: 512x512 -> 2048x2048)
    # -------------------------------------------------------------------------
    synthetic_lr = project_root / "samples" / "synthetic" / "synthetic_astro_input_512.png"
    synthetic_gt_path = project_root / "samples" / "synthetic" / "synthetic_astro_ground_truth_2048.png"
    catalog_path = project_root / "samples" / "synthetic" / "synthetic_ground_truth_catalog.json"

    with open(catalog_path, "r", encoding="utf-8") as f:
        gt_catalog = json.load(f)

    gt_arr = np.array(load_image(synthetic_gt_path))
    print(f"\n[DATASET 1] Expanded Synthetic Dataset ({gt_arr.shape[1]}x{gt_arr.shape[0]} GT)")
    print(f"  Ground Truth Stars:       {gt_catalog['groups']['Group_A']['total_stars']}")
    print(f"  Binary Pairs Included:    {gt_catalog['groups']['Group_A']['binary_pairs_count']}")
    print(f"  Cluster Stars Included:   {gt_catalog['groups']['Group_A']['cluster_stars_count']}")
    print(f"  Shadow Floor Intensity:   {gt_catalog['event_horizon']['shadow_floor_intensity']}")

    ablation_modes = [
        "baseline",
        "star_preserve_only",
        "filament_guard_only",
        "dark_sky_only",
        "star_plus_filament",
        "all_layers"
    ]

    synthetic_results: Dict[str, Any] = {}

    for mode in ablation_modes:
        out_img = exp_dir / f"synthetic_v011_{mode}.png"
        print(f"\n  >> Running Ablation Mode: [{mode}]...")

        res = pipeline.run(
            input_image=synthetic_lr,
            output_image=out_img,
            scale=4,
            tile_size=256,
            enable_astro_engine=(mode != "baseline"),
            ablation_mode=mode,
            astro_blend_weight=0.5
        )

        test_arr = np.array(load_image(out_img))
        psnr_gt = validator.compute_psnr(gt_arr, test_arr)
        ssim_detailed = validator.compute_ssim_detailed(gt_arr, test_arr)

        entry = {
            "mode": mode,
            "psnr_vs_ground_truth_db": psnr_gt,
            "ssim_y_vs_ground_truth": ssim_detailed["ssim_y"],
            "ssim_rgb_vs_ground_truth": ssim_detailed["ssim_rgb"],
            "profiling": res["profiling"],
            "output_dimensions": res["output_info"]["dimensions"],
            "output_size_mb": res["output_info"]["size_mb"]
        }

        if mode != "baseline":
            entry["star_preserve"] = res.get("star_preserve")
            entry["filament_guard"] = res.get("filament_guard")
            entry["dark_sky"] = res.get("dark_sky")
            entry["geometry_lock"] = res.get("geometry_lock")
            entry["fusion_stats"] = res.get("fusion_stats")
            entry["validation_vs_baseline"] = res.get("validation_metrics")

        synthetic_results[mode] = entry
        print(f"     PSNR vs GT: {psnr_gt:.2f} dB | SSIM Y: {ssim_detailed['ssim_y']:.4f} (RGB: {ssim_detailed['ssim_rgb']:.4f})")

    # -------------------------------------------------------------------------
    # PART 2: REAL ASTRONOMICAL TEST SCENE (deepzoom_astronomy_test.png, 682x1024 -> 2728x4096)
    # -------------------------------------------------------------------------
    astro_input = project_root / "samples" / "astronomical" / "deepzoom_astronomy_test.png"
    astro_baseline_out = exp_dir / "astronomy_test_4k_v011_baseline.png"
    astro_all_layers_out = exp_dir / "astronomy_test_4k_v011_all_layers.png"

    print(f"\n[DATASET 2] Real Astronomical Reference Image ({astro_input.name})")

    # A. Run baseline
    print("  >> Running Frozen Baseline Remacri...")
    res_astro_base = pipeline.run(
        input_image=astro_input,
        output_image=astro_baseline_out,
        scale=4,
        tile_size=400,
        enable_astro_engine=False,
        ablation_mode="baseline"
    )

    # B. Run full Astro Engine V0.1.1
    print("  >> Running Full DeepZoom Astro Engine V0.1.1...")
    res_astro_v011 = pipeline.run(
        input_image=astro_input,
        output_image=astro_all_layers_out,
        scale=4,
        tile_size=400,
        enable_astro_engine=True,
        ablation_mode="all_layers",
        astro_blend_weight=0.5
    )

    base_im_arr = np.array(load_image(astro_baseline_out))
    v011_im_arr = np.array(load_image(astro_all_layers_out))

    val_vs_baseline = validator.evaluate_all(base_im_arr, v011_im_arr, tile_size=400, scale=4)

    print(f"  Real Scene Output Master: {res_astro_v011['output_info']['dimensions']} ({res_astro_v011['output_info']['megapixels']} MP)")
    print(f"  PSNR vs Remacri Baseline: {val_vs_baseline['psnr_db']} dB")
    print(f"  SSIM vs Remacri Baseline: {val_vs_baseline['ssim']}")
    print(f"  Modified Pixels:          {val_vs_baseline['spatial_delta']['modified_pixel_pct']}%")
    print(f"  GPU Inference Time:       {res_astro_v011['profiling']['breakdown']['gpu_ncnn_inference_seconds']}s")
    print(f"  CPU Post-processing:      {res_astro_v011['profiling']['breakdown']['cpu_astro_postprocessing_seconds']}s")

    real_scene_report = {
        "input_image": str(astro_input.name),
        "target_resolution": res_astro_v011["output_info"]["dimensions"],
        "megapixels": res_astro_v011["output_info"]["megapixels"],
        "size_mb": res_astro_v011["output_info"]["size_mb"],
        "profiling": res_astro_v011["profiling"],
        "star_preserve": res_astro_v011["star_preserve"],
        "filament_guard": res_astro_v011["filament_guard"],
        "dark_sky": res_astro_v011["dark_sky"],
        "geometry_lock": res_astro_v011["geometry_lock"],
        "fusion_stats": res_astro_v011["fusion_stats"],
        "validation_against_baseline": val_vs_baseline
    }

    # -------------------------------------------------------------------------
    # PART 3: ASSEMBLE FULL REPORT & SAVE
    # -------------------------------------------------------------------------
    full_report = {
        "engine_version": "DeepZoom Astro Engine V0.1.1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": "AMD Radeon RX 5700 XT (8 GB VRAM, Vulkan device 0)",
            "cpu": "AMD Ryzen 5 9600X (6C/12T)",
            "ram": "16 GB RAM DDR5",
            "os": "Windows 11"
        },
        "audit_fixes_implemented": [
            "Local Gaussian-windowed SSIM (11x11, sigma=1.5, reflect padding)",
            "1-to-1 Hungarian star matching with strict distance gating (no zero drift bugs)",
            "Star classification: Point Sources vs Extended Sources vs Ambiguous",
            "Point source local PSF reconstruction: anti-ringing, centering, flux conservation",
            "Hessian eigenvector decomposition: tangential smoothing along v2, orthogonal anti-ringing along v1",
            "Elimination of arbitrary brightness multipliers in FilamentGuard",
            "Structure-aware vacuum stabilization in DarkSky: faint nebulosity never clipped",
            "Adaptive Otsu shadow basin segmentation in GeometryLock: handles I ~ 35-45 shadow floors",
            "Control-referenced tile seam analysis",
            "High-precision stage profiler separating GPU inference from CPU post-processing and I/O"
        ],
        "ablation_study_synthetic": synthetic_results,
        "real_astronomical_scene": real_scene_report
    }

    report_path = project_root / "benchmarks" / "astro_engine_v011_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[BENCHMARK COMPLETE] Full report successfully written to:")
    print(f"  {report_path}")
    print("=" * 80)
    return full_report

if __name__ == "__main__":
    run_v011_benchmark()
