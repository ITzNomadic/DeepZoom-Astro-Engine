"""
Test Suite & Benchmark for GeometryLock V0.1.2.
Validates:
1. GeneralGeometricValidator across known fiducials and spatial transformations.
2. BlackHoleShadowDetector across 8 stress scenarios:
   - Centered
   - Shifted Left
   - Shifted Right
   - Shifted Up
   - Shifted Down
   - Low Contrast
   - Non-zero Variable Floor (I=45)
   - Partially Occluded
Outputs benchmarks/v012/geometry_report.json.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from deepzoom.astro.geometry_lock.geometric_integrity import (
    GeometryLock,
    GeneralGeometricValidator,
    BlackHoleShadowDetector
)

def create_synthetic_scene(
    size: int = 300,
    center_offset: Tuple[int, int] = (0, 0),
    contrast_mult: float = 1.0,
    floor_intensity: float = 10.0,
    occlusion_angle_deg: float = 0.0,
    scale_from_base: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Creates a synthetic astronomical scene with fiducial control points and a configurable black hole shadow.
    """
    cx = size // 2 + center_offset[0]
    cy = size // 2 + center_offset[1]
    r_shadow = 35.0 * scale_from_base
    r_ring = 55.0 * scale_from_base
    w_ring = 8.0 * scale_from_base

    img = np.full((size, size), floor_intensity, dtype=np.float32)
    yy, xx = np.indices((size, size))
    dist = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)

    # Luminous accretion ring
    ring_dist = np.abs(dist - r_ring)
    ring_intensity = 180.0 * contrast_mult * np.exp(-(ring_dist ** 2) / (2.0 * w_ring ** 2))

    # Partial occlusion if requested
    if occlusion_angle_deg > 0:
        occ_rad = np.radians(occlusion_angle_deg)
        ring_intensity = np.where((phi >= -occ_rad/2) & (phi <= occ_rad/2), 0.0, ring_intensity)

    img += ring_intensity
    # Dark cavity
    img[dist <= r_shadow] = floor_intensity

    # Add 4 corner fiducial control points for general geometry
    margin = int(40 * scale_from_base)
    fiducials = np.array([
        [margin, margin],
        [size - margin, margin],
        [margin, size - margin],
        [size - margin, size - margin]
    ], dtype=np.float64)

    for fx, fy in fiducials:
        fx_i, fy_i = int(round(fx)), int(round(fy))
        half_w = max(int(2 * scale_from_base), 2)
        y0_f, y1_f = max(0, fy_i - half_w), min(size, fy_i + half_w + 1)
        x0_f, x1_f = max(0, fx_i - half_w), min(size, fx_i + half_w + 1)
        img[y0_f:y1_f, x0_f:x1_f] += 120.0

    meta = {
        "true_center": [cx, cy],
        "true_radius": r_shadow,
        "floor": floor_intensity,
        "contrast_mult": contrast_mult,
        "occlusion_deg": occlusion_angle_deg,
        "fiducials": fiducials.tolist()
    }
    return img, fiducials, meta


def run_geometry_suite():
    print("=" * 75)
    print("GEOMETRYLOCK V0.1.2 - COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 75)

    g_lock = GeometryLock()
    scale = 4.0

    scenarios = [
        {"name": "Centered_Ideal", "offset": (0, 0), "contrast": 1.0, "floor": 10.0, "occlusion": 0.0},
        {"name": "Shifted_Left", "offset": (-80, 0), "contrast": 1.0, "floor": 10.0, "occlusion": 0.0},
        {"name": "Shifted_Right", "offset": (80, 0), "contrast": 1.0, "floor": 10.0, "occlusion": 0.0},
        {"name": "Shifted_Up", "offset": (0, -70), "contrast": 1.0, "floor": 10.0, "occlusion": 0.0},
        {"name": "Shifted_Down", "offset": (0, 70), "contrast": 1.0, "floor": 10.0, "occlusion": 0.0},
        {"name": "Low_Contrast", "offset": (0, 0), "contrast": 0.25, "floor": 25.0, "occlusion": 0.0},
        {"name": "Variable_Floor_I45", "offset": (25, -25), "contrast": 1.0, "floor": 45.0, "occlusion": 0.0},
        {"name": "Partially_Occluded_150deg", "offset": (0, 0), "contrast": 1.0, "floor": 10.0, "occlusion": 150.0}
    ]

    results = []

    for sc in scenarios:
        name = sc["name"]
        print(f"\n[SCENARIO] {name}")
        img_in, fids_in, meta_in = create_synthetic_scene(
            size=300,
            center_offset=sc["offset"],
            contrast_mult=sc["contrast"],
            floor_intensity=sc["floor"],
            occlusion_angle_deg=sc["occlusion"]
        )

        # Generate exact scaled output (4x)
        size_hr = 300 * int(scale)
        offset_hr = (int(sc["offset"][0] * scale), int(sc["offset"][1] * scale))
        img_out, fids_out, meta_out = create_synthetic_scene(
            size=size_hr,
            center_offset=offset_hr,
            contrast_mult=sc["contrast"],
            floor_intensity=sc["floor"],
            occlusion_angle_deg=sc["occlusion"],
            scale_from_base=scale
        )

        eval_res = g_lock.evaluate_all(
            img_in_gray=img_in,
            img_out_gray=img_out,
            expected_scale=scale,
            control_points_in=fids_in,
            control_points_out=fids_out
        )

        gen_geom = eval_res["general_geometry"]
        sem_shad = eval_res["semantic_shadow_analysis"]

        print(f"  General Geometric Lock: {'PASS' if eval_res['geometric_lock_passed'] else 'FAIL'}")
        print(f"    Measured Scale:       {gen_geom['measured_scale']} (Err: {gen_geom['scale_error_pct']}%)")
        print(f"    RMSE Point Drift:     {gen_geom['rmse_point_drift_px']} px")
        print(f"  Semantic Shadow Detection:")
        print(f"    In Detected:          {sem_shad['input_shadow']['detected']} (Confidence: {sem_shad['input_shadow'].get('confidence')})")
        print(f"    Out Detected:         {sem_shad['output_shadow']['detected']} (Confidence: {sem_shad['output_shadow'].get('confidence')})")
        if sem_shad["shadow_tracking_verified"]:
            print(f"    Shadow Scale Error:   {sem_shad.get('shadow_scale_error_pct')}%")
            print(f"    Centroid Drift:       {sem_shad.get('centroid_drift_px')} px")
        else:
            print(f"    Note:                 {sem_shad.get('note', 'Shadow tracking not verified')}")

        results.append({
            "scenario": name,
            "parameters": sc,
            "general_lock_passed": eval_res["geometric_lock_passed"],
            "general_geometry": gen_geom,
            "semantic_shadow_analysis": sem_shad
        })

    report = {
        "suite": "GeometryLock_V0.1.2_Verification",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_scale": scale,
        "scenarios_evaluated": len(results),
        "all_general_locks_passed": all(r["general_lock_passed"] for r in results),
        "results": results
    }

    report_dir = project_root / "benchmarks" / "v012"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "geometry_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Geometry Report written to: {report_path}")
    print("=" * 75)
    return report

if __name__ == "__main__":
    from typing import Tuple
    run_geometry_suite()
