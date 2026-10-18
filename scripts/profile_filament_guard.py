"""
Instrumented Profiling Script for FilamentGuard V0.1.2 on 4K Master.
Measures time per operation, memory, and profiles CPU bottlenecks.
Outputs: benchmarks/v013/profiling_before.json
"""

import sys
import time
import json
import cProfile
import pstats
import io
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepzoom.astro.filament_guard.filament_analyzer import FilamentGuard


def profile_filament_guard_detailed():
    print("================================================================================")
    print("DEEPZOOM ASTRO ENGINE V0.1.3 - INSTRUMENTED FILAMENTGUARD PROFILING (BEFORE)")
    print("================================================================================")

    baseline_hr_path = PROJECT_ROOT / "output" / "experimental" / "v011" / "astronomy_test_4k_v011_baseline.png"
    if not baseline_hr_path.exists():
        raise FileNotFoundError(f"Missing baseline image: {baseline_hr_path}")

    print(f"Loading 4K image: {baseline_hr_path.name}")
    im = Image.open(baseline_hr_path)
    base_rgb = np.array(im)
    base_gray = np.array(im.convert("L"))
    h, w = base_gray.shape
    print(f"Dimensions: {w}x{h} ({h * w / 1e6:.2f} Megapixels)")

    fg = FilamentGuard(ridge_threshold=0.08, coherence_threshold=0.30, min_filament_length=15)

    timings = {}

    # Stage 1: Detailed breakdown of analyze_filaments
    t0 = time.perf_counter()
    arr = base_gray.astype(np.float32)
    timings["to_float32"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    import scipy.ndimage
    smoothed = scipy.ndimage.gaussian_filter(arr, sigma=1.0, mode='reflect')
    timings["gaussian_filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gy, gx = np.gradient(smoothed)
    timings["1st_gradient"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    gxy_sym = 0.5 * (gyx + gxy)
    timings["2nd_gradient"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    trace = gxx + gyy
    det = (gxx * gyy) - (gxy_sym ** 2)
    diff = np.sqrt(np.maximum((trace / 2.0) ** 2 - det, 0.0))
    e1 = (trace / 2.0) + diff
    e2 = (trace / 2.0) - diff
    swap_mask = np.abs(e2) > np.abs(e1)
    lambda1 = np.where(swap_mask, e2, e1)
    lambda2 = np.where(swap_mask, e1, e2)
    timings["eigenvalues"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vy = gxx - lambda1
    vx = -gxy_sym
    norm = np.hypot(vy, vx)
    fallback_mask = norm < 1e-5
    vy2 = np.where(fallback_mask, -gxy_sym, vy)
    vx2 = np.where(fallback_mask, gyy - lambda1, vx)
    norm2 = np.maximum(np.hypot(vy2, vx2), 1e-6)
    v1_y = vy2 / norm2
    v1_x = vx2 / norm2
    v2_y = -v1_x
    v2_x = v1_y
    v1 = np.stack([v1_y, v1_x], axis=-1)
    v2 = np.stack([v2_y, v2_x], axis=-1)
    timings["eigenvectors"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ridge_strength = np.maximum(-lambda1, 0.0)
    p99 = float(np.percentile(ridge_strength, 99.0))
    norm_factor = max(p99, 1e-4)
    normalized_ridges = np.clip(ridge_strength / norm_factor, 0.0, 1.0)
    coherence_num = (lambda1 - lambda2) ** 2
    coherence_den = (lambda1 + lambda2) ** 2 + 1e-4
    coherence = np.clip(coherence_num / coherence_den, 0.0, 1.0)
    raw_mask = (normalized_ridges > 0.08) & (coherence > 0.30)
    timings["ridge_and_coherence_maps"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    labeled, num_features = scipy.ndimage.label(raw_mask)
    timings["scipy_label"] = time.perf_counter() - t0
    print(f"  Raw features labeled: {num_features}")

    t0 = time.perf_counter()
    if num_features > 0:
        sizes = scipy.ndimage.sum(raw_mask, labeled, range(1, num_features + 1))
        large_labels = np.where(sizes >= 15)[0] + 1
        t_sum = time.perf_counter() - t0
        timings["scipy_sum_sizes"] = t_sum
        
        t0_isin = time.perf_counter()
        filament_mask = np.isin(labeled, large_labels)
        timings["np_isin_filtering"] = time.perf_counter() - t0_isin
    else:
        filament_mask = raw_mask
        timings["scipy_sum_sizes"] = 0.0
        timings["np_isin_filtering"] = 0.0

    t0 = time.perf_counter()
    gy_raw, gx_raw = np.gradient(arr)
    grad_mag = np.hypot(gx_raw, gy_raw)
    laplacian = lambda1 + lambda2
    p85_grad = float(np.percentile(grad_mag, 85.0))
    med_lap = float(np.median(np.abs(laplacian)))
    overshoot_mask = (laplacian > (1.8 * max(med_lap, 1e-4)))
    high_grad_mask = (grad_mag > p85_grad)
    ringing_index = float(np.mean(overshoot_mask[high_grad_mask])) if np.any(high_grad_mask) else 0.0
    timings["ringing_detection"] = time.perf_counter() - t0

    analysis = {
        "filament_coverage_pct": float(np.mean(filament_mask) * 100.0),
        "mean_filament_coherence": float(np.mean(coherence[filament_mask])) if np.any(filament_mask) else 0.0,
        "ringing_index": ringing_index,
        "ringing_detected": bool(ringing_index > 0.22),
        "normalized_ridge_map": normalized_ridges,
        "coherence_map": coherence,
        "filament_mask": filament_mask,
        "v1": v1,
        "v2": v2,
        "lambda1": lambda1,
        "lambda2": lambda2
    }

    # Stage 2: Detailed breakdown of repair_filaments
    t0_rep = time.perf_counter()
    arr_rgb = base_rgb.astype(np.float32)
    h, w, c = arr_rgb.shape

    t0 = time.perf_counter()
    yy, xx = np.indices((h, w))
    step_t = 1.5
    t_plus_y = np.clip(np.round(yy + step_t * v2[..., 0]).astype(int), 0, h - 1)
    t_plus_x = np.clip(np.round(xx + step_t * v2[..., 1]).astype(int), 0, w - 1)
    t_minus_y = np.clip(np.round(yy - step_t * v2[..., 0]).astype(int), 0, h - 1)
    t_minus_x = np.clip(np.round(xx - step_t * v2[..., 1]).astype(int), 0, w - 1)
    step_n_amb = 3.5
    na_plus_y = np.clip(np.round(yy + step_n_amb * v1[..., 0]).astype(int), 0, h - 1)
    na_plus_x = np.clip(np.round(xx + step_n_amb * v1[..., 1]).astype(int), 0, w - 1)
    na_minus_y = np.clip(np.round(yy - step_n_amb * v1[..., 0]).astype(int), 0, h - 1)
    na_minus_x = np.clip(np.round(xx - step_n_amb * v1[..., 1]).astype(int), 0, w - 1)
    timings["coordinate_grid_generation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    flank_zone = scipy.ndimage.binary_dilation(filament_mask, iterations=3) & (~filament_mask)
    timings["flank_zone_dilation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    repaired = arr_rgb.copy()
    for ch in range(c):
        channel = arr_rgb[..., ch]
        tangential_interp = 0.5 * (channel[t_plus_y, t_plus_x] + channel[t_minus_y, t_minus_x])
        ridge_healed = np.maximum(channel, tangential_interp)
        ambient_floor = np.minimum(channel[na_plus_y, na_plus_x], channel[na_minus_y, na_minus_x])
        trough_repaired = np.maximum(channel, ambient_floor)
        repaired_channel = channel.copy()
        repaired_channel = np.where(filament_mask, ridge_healed, repaired_channel)
        repaired_channel = np.where(flank_zone & (channel < ambient_floor), trough_repaired, repaired_channel)
        repaired[..., ch] = repaired_channel
    timings["directional_sampling_and_where"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    labeled_filaments, num_frags = scipy.ndimage.label(filament_mask)
    from scipy.spatial import cKDTree
    gray_for_bridge = 0.299 * arr_rgb[..., 0] + 0.587 * arr_rgb[..., 1] + 0.114 * arr_rgb[..., 2]
    slices = scipy.ndimage.find_objects(labeled_filaments)
    frag_core_points = []
    frag_labels = []
    for idx, slc in enumerate(slices):
        if slc is None:
            continue
        lbl = idx + 1
        sub_mask = (labeled_filaments[slc] == lbl)
        if not np.any(sub_mask):
            continue
        sub_gray = gray_for_bridge[slc]
        max_val = np.max(sub_gray[sub_mask])
        th = max(max_val * 0.5, 30.0)
        sub_core = sub_mask & (sub_gray >= th)
        if not np.any(sub_core):
            sub_core = sub_mask
        core_y, core_x = np.where(sub_core)
        core_y = core_y + slc[0].start
        core_x = core_x + slc[1].start
        pts = np.column_stack((core_y, core_x))
        frag_core_points.append(pts)
        frag_labels.append(np.full(len(pts), lbl, dtype=np.int32))
    
    if frag_core_points:
        all_pts = np.vstack(frag_core_points)
        all_lbls = np.concatenate(frag_labels)
        tree = cKDTree(all_pts)
        pairs = tree.query_pairs(r=12.0)
        best_bridge_per_pair = {}
        for idx1, idx2 in pairs:
            l1 = int(all_lbls[idx1])
            l2 = int(all_lbls[idx2])
            if l1 != l2:
                p1 = all_pts[idx1]
                p2 = all_pts[idx2]
                d = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
                pair_key = (min(l1, l2), max(l1, l2))
                if pair_key not in best_bridge_per_pair or d < best_bridge_per_pair[pair_key][0]:
                    best_bridge_per_pair[pair_key] = (d, p1, p2)
        for (l1, l2), (min_d, c1, c2) in best_bridge_per_pair.items():
            steps = np.linspace(0.0, 1.0, int(round(min_d * 4)) + 5)
            for ch in range(c):
                val1 = float(arr_rgb[c1[0], c1[1], ch])
                val2 = float(arr_rgb[c2[0], c2[1], ch])
                for t in steps:
                    pt = (1.0 - t) * c1 + t * c2
                    val = (1.0 - t) * val1 + t * val2
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            yi = int(round(pt[0] + dy))
                            xi = int(round(pt[1] + dx))
                            if 0 <= yi < h and 0 <= xi < w:
                                d = np.hypot(yi - pt[0], xi - pt[1])
                                w_pt = np.exp(-0.5 * (d / 1.5) ** 2)
                                repaired[yi, xi, ch] = max(repaired[yi, xi, ch], val * w_pt)
    timings["fragment_bridging"] = time.perf_counter() - t0

    repair_mask = np.zeros((h, w), dtype=np.float32)
    repair_mask[filament_mask] = np.clip(coherence[filament_mask] * normalized_ridges[filament_mask], 0.0, 1.0)
    repair_mask[flank_zone] = np.clip(0.5 * coherence[flank_zone] * normalized_ridges[flank_zone], 0.0, 0.5)

    total_time = sum(timings.values())
    print("\n--- TIMING BREAKDOWN ---")
    for k, v in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:35s}: {v:8.3f} s ({v / total_time * 100:5.1f}%)")
    print(f"  TOTAL MEASURED TIME:               {total_time:8.3f} s")

    out_json = {
        "version": "0.1.2_unoptimized",
        "image_dimensions": [w, h],
        "megapixels": round(h * w / 1e6, 2),
        "total_time_seconds": round(total_time, 3),
        "timings_breakdown_seconds": {k: round(v, 4) for k, v in timings.items()},
        "percentages": {k: round(v / total_time * 100, 2) for k, v in timings.items()}
    }

    out_dir = PROJECT_ROOT / "benchmarks" / "v013"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "profiling_before.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved profiling report to: {out_path}")


if __name__ == "__main__":
    profile_filament_guard_detailed()
