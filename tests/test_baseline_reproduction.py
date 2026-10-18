"""
Baseline Reproduction & Determinism Verification Test.
Compares DeepZoomStudio migrated engine output against the original TikTokAIStudio reference.
"""

import sys
import json
import time
from pathlib import Path

# Add src to sys.path dynamically
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from deepzoom.core.config import DeepZoomConfig
from deepzoom.inference.ncnn_engine import NCNNEngine
from deepzoom.utils.image_io import calculate_sha256, compare_images_pixelwise, get_image_info

def run_baseline_reproduction_test():
    config = DeepZoomConfig()
    engine = NCNNEngine(config)

    print("=" * 70)
    print("DEEPZOOM STUDIO - BASELINE REPRODUCTION VERIFICATION")
    print("=" * 70)
    print(f"Project Root:    {config.root_dir}")
    print(f"NCNN Runtime:    {config.ncnn_runtime_path}")
    print(f"Models Dir:      {config.models_dir}")
    print(f"Target GPU:      {config.default_gpu_id}")
    print("-" * 70)

    # TEST 1: Strategic Crop (512x512 -> 2048x2048) Photon Ring
    crop_input = config.root_dir / "benchmarks" / "crops" / "base_crop1_photon_ring.png"
    crop_reference = config.root_dir / "benchmarks" / "crops" / "base_crop1_photon_ring_remacri-4x.png"
    crop_output = config.root_dir / "output" / "baseline" / "reproduced_crop1_remacri.png"

    print("\n[TEST 1] Reproducing Strategic Crop 1 (Photon Ring, 512x512 -> 2048x2048)...")
    t0 = time.time()
    res1 = engine.upscale(
        input_image=crop_input,
        output_image=crop_output,
        model_name="remacri-4x",
        scale=4,
        tile_size=256,
        gpu_id=0
    )
    t1 = time.time()
    print(f"  Execution finished in {res1['execution_time_seconds']}s")

    comp1 = compare_images_pixelwise(crop_output, crop_reference)
    hash_out1 = calculate_sha256(crop_output)
    hash_ref1 = calculate_sha256(crop_reference)

    print(f"  Output Hash:    {hash_out1}")
    print(f"  Reference Hash: {hash_ref1}")
    print(f"  Exact Pixel Match: {comp1['exact_match']}")
    print(f"  Mean Abs Diff:     {comp1['mean_abs_diff']}")
    print(f"  Max Diff:          {comp1['max_diff']}")
    print(f"  Matching Pixels:   {comp1['matching_pixels_pct']:.2f}%")

    # TEST 2: Full Master 8K (1152x2048 -> 4608x8192)
    full_input = config.root_dir / "samples" / "astronomical" / "black_hole_base_1152x2048.png"
    full_reference = config.root_dir / "assets" / "reference" / "results_4k_8k" / "blackhole_8k_4608x8192_remacri.png"
    full_output = config.root_dir / "output" / "baseline" / "reproduced_blackhole_8k_remacri.png"

    print("\n[TEST 2] Reproducing Full Master 8K (1152x2048 -> 4608x8192, 37.75 MP)...")
    t0 = time.time()
    res2 = engine.upscale(
        input_image=full_input,
        output_image=full_output,
        model_name="remacri-4x",
        scale=4,
        tile_size=256,
        gpu_id=0
    )
    t1 = time.time()
    print(f"  Execution finished in {res2['execution_time_seconds']}s")

    comp2 = compare_images_pixelwise(full_output, full_reference)
    hash_out2 = calculate_sha256(full_output)
    hash_ref2 = calculate_sha256(full_reference)

    print(f"  Output Hash:    {hash_out2}")
    print(f"  Reference Hash: {hash_ref2}")
    print(f"  Exact Pixel Match: {comp2['exact_match']}")
    print(f"  Mean Abs Diff:     {comp2['mean_abs_diff']}")
    print(f"  Max Diff:          {comp2['max_diff']}")
    print(f"  Matching Pixels:   {comp2['matching_pixels_pct']:.2f}%")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "AMD Radeon RX 5700 XT (GPU 0)",
        "test1_crop": {
            "dimensions": comp1["dimensions"],
            "exact_match": comp1["exact_match"],
            "mean_abs_diff": comp1["mean_abs_diff"],
            "max_diff": comp1["max_diff"],
            "matching_pixels_pct": comp1["matching_pixels_pct"],
            "hash_match": bool(hash_out1 == hash_ref1),
            "output_hash": hash_out1,
            "reference_hash": hash_ref1
        },
        "test2_full_8k": {
            "dimensions": comp2["dimensions"],
            "exact_match": comp2["exact_match"],
            "mean_abs_diff": comp2["mean_abs_diff"],
            "max_diff": comp2["max_diff"],
            "matching_pixels_pct": comp2["matching_pixels_pct"],
            "hash_match": bool(hash_out2 == hash_ref2),
            "output_hash": hash_out2,
            "reference_hash": hash_ref2
        }
    }

    report_path = config.root_dir / "benchmarks" / "baseline_reproduction_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Results written to: {report_path}")
    print("=" * 70)
    return report

if __name__ == "__main__":
    run_baseline_reproduction_test()
