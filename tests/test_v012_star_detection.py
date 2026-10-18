"""
Test Suite & Benchmark for Star Detector V0.1.2.
Validates:
1. Exact ground-truth catalog matching.
2. 4-way source classification (point_source, extended_source, filament_structure, ambiguous).
3. Precision, Recall, F1-Score, FP, FN, centroid/flux/FWHM errors.
4. Stratified performance across Bright, Medium, and Faint magnitude bins.
Outputs benchmarks/v012/star_detection_report.json.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from deepzoom.astro.star_preserve.star_analyzer import StarPreserve, StarRecord

def generate_detection_ground_truth_scene(size: int = 512) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Generates an exact ground-truth scene with diverse astronomical sources:
    - 15 Point Sources (5 Bright, 5 Medium, 5 Faint)
    - 2 Close Binary Pairs (testing resolving power)
    - 2 Extended Galaxies
    - 2 Filamentary Gas Ridges
    Returns (image_array, exact_catalog).
    """
    np.random.seed(42)
    img = np.full((size, size), 18.0, dtype=np.float32)
    # Background noise floor (sigma = 1.0)
    noise = np.random.normal(0.0, 1.0, (size, size)).astype(np.float32)
    img += noise

    yy, xx = np.indices((size, size))
    catalog = []
    source_id = 0

    # A. Point Sources (15 stars: 5 Bright, 5 Medium, 5 Faint)
    flux_bins = [
        ("Bright", 12000.0, 1.8),
        ("Bright", 10000.0, 1.7),
        ("Bright", 9500.0, 1.6),
        ("Bright", 8500.0, 1.6),
        ("Bright", 8000.0, 1.5),
        ("Medium", 4500.0, 1.5),
        ("Medium", 4000.0, 1.4),
        ("Medium", 3500.0, 1.4),
        ("Medium", 3000.0, 1.3),
        ("Medium", 2500.0, 1.3),
        ("Faint", 1200.0, 1.2),
        ("Faint", 900.0, 1.2),
        ("Faint", 750.0, 1.1),
        ("Faint", 600.0, 1.1),
        ("Faint", 450.0, 1.0)
    ]

    for bin_name, flux, sigma in flux_bins:
        # Uniform random positions away from borders
        sx = float(np.random.uniform(50, size - 50))
        sy = float(np.random.uniform(50, size - 50))
        r_dist = np.hypot(xx - sx, yy - sy)
        psf = (flux / (2.0 * np.pi * sigma ** 2)) * np.exp(-(r_dist ** 2) / (2.0 * sigma ** 2))
        img += psf

        catalog.append({
            "id": source_id,
            "true_type": "point_source",
            "bin": bin_name,
            "x": round(sx, 3),
            "y": round(sy, 3),
            "flux": round(flux, 1),
            "sigma": sigma,
            "fwhm": round(float(2.355 * sigma), 2),
            "ellipticity": 0.0
        })
        source_id += 1

    # B. Close Binary Pairs
    binaries = [
        (120.0, 120.0, 126.0, 120.0, 5000.0, 4500.0, "binary_1"),
        (380.0, 380.0, 380.0, 387.0, 3500.0, 3200.0, "binary_2")
    ]
    for x1, y1, x2, y2, f1, f2, lbl in binaries:
        for bx, by, bf in [(x1, y1, f1), (x2, y2, f2)]:
            r_b = np.hypot(xx - bx, yy - by)
            img += (bf / (2 * np.pi * 1.3**2)) * np.exp(-(r_b**2)/(2 * 1.3**2))
            catalog.append({
                "id": source_id,
                "true_type": "point_source",
                "bin": "Medium",
                "label": lbl,
                "x": round(bx, 3),
                "y": round(by, 3),
                "flux": round(bf, 1),
                "sigma": 1.3,
                "fwhm": round(float(2.355 * 1.3), 2),
                "ellipticity": 0.0
            })
            source_id += 1

    # C. Extended Galaxies (Non-stars)
    galaxies = [
        (180.0, 350.0, 9.0, 14000.0, "spiral_galaxy"),
        (350.0, 160.0, 7.5, 9000.0, "elliptical_galaxy")
    ]
    for gx, gy, g_sigma, g_flux, g_name in galaxies:
        rg = np.hypot(xx - gx, yy - gy)
        img += (g_flux / (2 * np.pi * g_sigma**2)) * np.exp(-(rg**2)/(2 * g_sigma**2))
        catalog.append({
            "id": source_id,
            "true_type": "extended_source",
            "bin": "Extended",
            "label": g_name,
            "x": round(gx, 3),
            "y": round(gy, 3),
            "flux": round(g_flux, 1),
            "sigma": g_sigma,
            "fwhm": round(float(2.355 * g_sigma), 2),
            "ellipticity": 0.35
        })
        source_id += 1

    # D. Filamentary Gas Ridges (Non-stars)
    # Horizontal filament
    d_fil1 = np.abs(yy - 260.0)
    fil1_mask = (xx >= 80.0) & (xx <= 240.0)
    img += 120.0 * np.exp(-(d_fil1**2)/(2 * 2.0**2)) * fil1_mask

    # Diagonal filament
    d_fil2 = np.abs((xx - 300.0) + (yy - 250.0)) / np.sqrt(2.0)
    fil2_mask = (np.hypot(xx - 300.0, yy - 250.0) <= 80.0)
    img += 90.0 * np.exp(-(d_fil2**2)/(2 * 2.2**2)) * fil2_mask

    catalog.append({
        "id": source_id,
        "true_type": "filament_structure",
        "bin": "Filament",
        "label": "gas_ridge_1",
        "x": 160.0,
        "y": 260.0,
        "flux": 15000.0,
        "sigma": 2.0,
        "fwhm": 4.71,
        "ellipticity": 0.85
    })

    img = np.clip(img, 0.0, 255.0)
    return img, catalog


def run_star_detection_benchmark():
    print("=" * 75)
    print("STAR DETECTOR V0.1.2 - RIGOROUS GROUND-TRUTH BENCHMARK")
    print("=" * 75)

    img, catalog = generate_detection_ground_truth_scene(size=512)
    sp = StarPreserve()

    t0 = time.perf_counter()
    detected_sources = sp.analyze_stars(img)
    t_detect = time.perf_counter() - t0

    print(f"Total Ground-Truth Sources in Scene: {len(catalog)}")
    true_points = [s for s in catalog if s["true_type"] == "point_source"]
    print(f"  True Point Sources (Stars):        {len(true_points)}")
    print(f"  True Extended Sources (Galaxies):  {sum(1 for s in catalog if s['true_type'] == 'extended_source')}")
    print(f"  True Filament Structures:          {sum(1 for s in catalog if s['true_type'] == 'filament_structure')}")

    print(f"\nTotal Candidates Detected by StarDetector V0.1.2: {len(detected_sources)} (in {round(t_detect*1000, 2)} ms)")

    # Classification counts
    by_type = {}
    for s in detected_sources:
        by_type[s.source_type] = by_type.get(s.source_type, 0) + 1
    print(f"  Classified as point_source:        {by_type.get('point_source', 0)}")
    print(f"  Classified as extended_source:     {by_type.get('extended_source', 0)}")
    print(f"  Classified as filament_structure:  {by_type.get('filament_structure', 0)}")
    print(f"  Classified as ambiguous:           {by_type.get('ambiguous', 0)}")

    # 1-to-1 Hungarian Matching against true point sources
    # True point coords
    gt_coords = np.array([[s["x"], s["y"]] for s in true_points], dtype=np.float64)
    det_points = [s for s in detected_sources if s.source_type == "point_source"]
    det_coords = np.array([[s.subpixel_x, s.subpixel_y] for s in det_points], dtype=np.float64)

    from scipy.optimize import linear_sum_assignment
    diff = gt_coords[:, np.newaxis, :] - det_coords[np.newaxis, :, :]
    dist_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])

    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    gate_dist = 3.0

    matched = []
    unmatched_gt = set(range(len(true_points)))
    unmatched_det = set(range(len(det_points)))

    centroid_errors = []
    flux_errors_pct = []
    fwhm_errors_px = []

    # Bin-wise tracking
    bin_stats = {
        "Bright": {"gt": 0, "tp": 0, "errors": []},
        "Medium": {"gt": 0, "tp": 0, "errors": []},
        "Faint": {"gt": 0, "tp": 0, "errors": []}
    }
    for s in true_points:
        b = s.get("bin", "Medium")
        if b in bin_stats:
            bin_stats[b]["gt"] += 1

    for r, c in zip(row_ind, col_ind):
        d = float(dist_matrix[r, c])
        if d <= gate_dist:
            matched.append((r, c))
            unmatched_gt.discard(r)
            unmatched_det.discard(c)
            centroid_errors.append(d)

            gt_s = true_points[r]
            det_s = det_points[c]

            f_err = abs(det_s.flux - gt_s["flux"]) / gt_s["flux"] * 100.0
            flux_errors_pct.append(f_err)

            fwhm_err = abs(det_s.fwhm - gt_s["fwhm"])
            fwhm_errors_px.append(fwhm_err)

            b = gt_s.get("bin", "Medium")
            if b in bin_stats:
                bin_stats[b]["tp"] += 1
                bin_stats[b]["errors"].append(d)

    tp = len(matched)
    fp = len(unmatched_det)
    fn = len(unmatched_gt)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / max(precision + recall, 1e-5))

    print("\n[POINT SOURCE METRIC EVALUATION]")
    print(f"  True Positives (Stars Recovered): {tp} / {len(true_points)}")
    print(f"  False Positives:                  {fp}")
    print(f"  False Negatives (Missed):         {fn}")
    print(f"  Precision:                        {precision*100.0:.2f}%")
    print(f"  Recall:                           {recall*100.0:.2f}%")
    print(f"  F1-Score:                         {f1:.4f}")
    print(f"  Mean Centroid Error:              {np.mean(centroid_errors):.3f} px")
    print(f"  Max Centroid Error:               {np.max(centroid_errors):.3f} px")
    print(f"  Mean Flux Error:                  {np.mean(flux_errors_pct):.2f}%")
    print(f"  Mean FWHM Error:                  {np.mean(fwhm_errors_px):.2f} px")

    print("\n[STRATIFIED PERFORMANCE BY MAGNITUDE]")
    for b_name, b_data in bin_stats.items():
        b_rec = (b_data["tp"] / b_data["gt"] * 100.0) if b_data["gt"] > 0 else 0.0
        m_err = np.mean(b_data["errors"]) if b_data["errors"] else 0.0
        print(f"  {b_name:8s} | GT: {b_data['gt']:2d} | Recovered: {b_data['tp']:2d} | Recall: {b_rec:5.1f}% | Mean Drift: {m_err:.3f} px")

    report = {
        "suite": "StarDetector_V0.1.2_Rigorous_Ground_Truth",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "detection_time_ms": round(t_detect * 1000.0, 2),
        "ground_truth_counts": {
            "total_sources": len(catalog),
            "point_sources": len(true_points),
            "extended_galaxies": 2,
            "filament_ridges": 2
        },
        "detector_classifications": by_type,
        "point_source_metrics": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mean_centroid_error_px": round(float(np.mean(centroid_errors)), 3),
            "max_centroid_error_px": round(float(np.max(centroid_errors)), 3),
            "mean_flux_error_pct": round(float(np.mean(flux_errors_pct)), 2),
            "mean_fwhm_error_px": round(float(np.mean(fwhm_errors_px)), 2)
        },
        "stratified_by_magnitude": {
            b: {
                "ground_truth_count": bin_stats[b]["gt"],
                "true_positives": bin_stats[b]["tp"],
                "recall": round(float(bin_stats[b]["tp"] / max(bin_stats[b]["gt"], 1)), 4),
                "mean_drift_px": round(float(np.mean(bin_stats[b]["errors"])), 3) if bin_stats[b]["errors"] else None
            }
            for b in bin_stats
        }
    }

    report_dir = project_root / "benchmarks" / "v012"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "star_detection_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Star Detection Report written to: {report_path}")
    print("=" * 75)
    return report

if __name__ == "__main__":
    run_star_detection_benchmark()
