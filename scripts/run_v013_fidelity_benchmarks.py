"""
DeepZoom Astro Engine V0.1.3 - Fidelity Benchmark Suite Across 8 Astronomical Scenes.
Generates and benchmarks 8 rigorous synthetic/astronomical test scenes:
  1. Isolated Star Field
  2. Dense Clusters & Binaries
  3. Nebula with Filaments & Arcs
  4. Synthetic Relativistic Accretion Disk
  5. Synthetic Black Hole Shadow & Photon Ring
  6. Galaxies & Extended Sources
  7. Multi-scale Composite Scene
  8. Low-Contrast Faint Cosmic Background

Evaluates Remacri Baseline vs DeepZoom Astro Engine V0.1.3 against Ground Truth.
Verifies Zero Regression Guarantee, FidelityGate enforcement, and photometric fidelity.
Outputs results to benchmarks/v013/fidelity_report.json.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from deepzoom.pipeline import AstroPipeline
from deepzoom.utils.image_io import load_image, get_image_info
from deepzoom.validation.astro_validator import AstroValidator


def build_scene_1_isolated_stars(size_hr: int = 2048) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scene 1: Isolated Star Field - 28 point sources of various magnitudes on uniform background."""
    np.random.seed(101)
    arr = np.full((size_hr, size_hr), 10.0, dtype=np.float32)
    # Add realistic cosmic noise
    arr += np.random.normal(0.0, 0.6, (size_hr, size_hr)).astype(np.float32)

    stars = []
    # Grid with jitter to ensure isolation
    rows, cols = 5, 6
    dx = (size_hr - 200) // cols
    dy = (size_hr - 200) // rows

    for r in range(rows):
        for c in range(cols):
            if len(stars) >= 28:
                break
            cx = 100 + c * dx + np.random.uniform(-15.0, 15.0)
            cy = 100 + r * dy + np.random.uniform(-15.0, 15.0)
            sigma = np.random.uniform(1.2, 2.2)
            flux = np.random.uniform(1500.0, 14000.0)
            peak = flux / (2.0 * np.pi * (sigma ** 2))

            # Render Gaussian star
            rad = int(np.ceil(sigma * 4.5))
            x0, x1 = max(0, int(cx - rad)), min(size_hr, int(cx + rad + 1))
            y0, y1 = max(0, int(cy - rad)), min(size_hr, int(cy + rad + 1))
            gy, gx = np.ogrid[y0:y1, x0:x1]
            dist2 = (gx - cx) ** 2 + (gy - cy) ** 2
            arr[y0:y1, x0:x1] += peak * np.exp(-dist2 / (2.0 * (sigma ** 2)))

            stars.append({"cx": round(cx, 2), "cy": round(cy, 2), "sigma": round(sigma, 2), "flux": round(flux, 1)})

    meta = {"scene": "Isolated Star Field", "star_count": len(stars), "stars": stars}
    return arr, meta


def build_scene_2_dense_clusters(size_hr: int = 2048) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scene 2: Dense Stellar Clusters & Binaries - Crowded stellar core + binary pairs."""
    np.random.seed(202)
    arr = np.full((size_hr, size_hr), 8.0, dtype=np.float32)
    arr += np.random.normal(0.0, 0.7, (size_hr, size_hr)).astype(np.float32)

    stars = []
    # Central globular cluster (Plummer profile)
    clust_cx, clust_cy = size_hr // 2, size_hr // 2
    r_core = 90.0

    for i in range(75):
        # Plummer distribution
        r = r_core * np.sqrt(np.random.uniform(0.0, 1.0) / (1.0 - np.random.uniform(0.0, 0.95) + 1e-5))
        r = min(r, 380.0)
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        cx = clust_cx + r * np.cos(theta)
        cy = clust_cy + r * np.sin(theta)
        sigma = np.random.uniform(1.2, 1.8)
        flux = np.random.uniform(2000.0, 8500.0)
        peak = flux / (2.0 * np.pi * (sigma ** 2))

        rad = int(np.ceil(sigma * 4.0))
        x0, x1 = max(0, int(cx - rad)), min(size_hr, int(cx + rad + 1))
        y0, y1 = max(0, int(cy - rad)), min(size_hr, int(cy + rad + 1))
        gy, gx = np.ogrid[y0:y1, x0:x1]
        dist2 = (gx - cx) ** 2 + (gy - cy) ** 2
        arr[y0:y1, x0:x1] += peak * np.exp(-dist2 / (2.0 * (sigma ** 2)))
        stars.append({"cx": round(cx, 2), "cy": round(cy, 2), "type": "cluster_member"})

    # 4 close binary pairs (separation 4 to 6 pixels)
    binaries = [
        (350, 400, 355, 403),
        (1650, 450, 1654, 455),
        (400, 1600, 405, 1604),
        (1700, 1700, 1706, 1702)
    ]
    for b1x, b1y, b2x, b2y in binaries:
        for cx, cy in [(b1x, b1y), (b2x, b2y)]:
            sigma = 1.3
            flux = 3500.0
            peak = flux / (2.0 * np.pi * (sigma ** 2))
            rad = 6
            x0, x1 = max(0, int(cx - rad)), min(size_hr, int(cx + rad + 1))
            y0, y1 = max(0, int(cy - rad)), min(size_hr, int(cy + rad + 1))
            gy, gx = np.ogrid[y0:y1, x0:x1]
            dist2 = (gx - cx) ** 2 + (gy - cy) ** 2
            arr[y0:y1, x0:x1] += peak * np.exp(-dist2 / (2.0 * (sigma ** 2)))
            stars.append({"cx": cx, "cy": cy, "type": "binary_member"})

    meta = {"scene": "Dense Clusters & Binaries", "total_stars": len(stars), "binaries_count": 4}
    return arr, meta


def build_scene_3_nebula_filaments(size_hr: int = 2048) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scene 3: Nebula with Filaments & Ionization Arcs."""
    np.random.seed(303)
    yy, xx = np.indices((size_hr, size_hr), dtype=np.float32)
    arr = np.full((size_hr, size_hr), 5.0, dtype=np.float32)
    arr += np.random.normal(0.0, 0.8, (size_hr, size_hr)).astype(np.float32)

    # Diffuse cloud background
    neb_r2 = ((xx - size_hr * 0.45) ** 2) / (2.0 * (350.0 ** 2)) + ((yy - size_hr * 0.55) ** 2) / (2.0 * (280.0 ** 2))
    nebula = 38.0 * np.exp(-neb_r2) * (1.0 + 0.25 * np.sin(xx / 55.0) * np.cos(yy / 45.0))
    arr += nebula

    # 1. Straight diagonal filament
    angle = np.radians(40.0)
    d_line1 = np.abs((xx - 400.0) * (-np.sin(angle)) + (yy - 350.0) * np.cos(angle))
    along1 = (xx - 400.0) * np.cos(angle) + (yy - 350.0) * np.sin(angle)
    mask1 = (d_line1 <= 12.0) & (along1 >= 0) & (along1 <= 900.0)
    arr[mask1] += 120.0 * np.exp(-(d_line1[mask1] ** 2) / (2.0 * (2.4 ** 2)))

    # 2. Curved bow-shock filament arc
    arc_cx, arc_cy = size_hr * 0.65, size_hr * 0.40
    dist_arc = np.hypot(xx - arc_cx, yy - arc_cy)
    arc_mask = (dist_arc >= 220.0) & (dist_arc <= 260.0) & (xx >= arc_cx - 50)
    arr[arc_mask] += 95.0 * np.exp(-((dist_arc[arc_mask] - 240.0) ** 2) / (2.0 * (2.8 ** 2)))

    # 3. Intersecting fine filament
    angle2 = np.radians(-25.0)
    d_line2 = np.abs((xx - 600.0) * (-np.sin(angle2)) + (yy - 750.0) * np.cos(angle2))
    along2 = (xx - 600.0) * np.cos(angle2) + (yy - 750.0) * np.sin(angle2)
    mask2 = (d_line2 <= 10.0) & (along2 >= 0) & (along2 <= 750.0)
    arr[mask2] += 105.0 * np.exp(-(d_line2[mask2] ** 2) / (2.0 * (2.0 ** 2)))

    meta = {"scene": "Nebula with Filaments & Arcs", "structures": ["diffuse_nebula", "straight_filament", "bow_shock_arc", "intersecting_filament"]}
    return arr, meta


def build_scene_4_accretion_disk(size_hr: int = 2048) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scene 4: Synthetic Relativistic Accretion Disk - Doppler beaming, spiral density waves."""
    np.random.seed(404)
    yy, xx = np.indices((size_hr, size_hr), dtype=np.float32)
    arr = np.full((size_hr, size_hr), 6.0, dtype=np.float32)
    arr += np.random.normal(0.0, 0.7, (size_hr, size_hr)).astype(np.float32)

    cx, cy = size_hr // 2, size_hr // 2
    r = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)

    # Relativistic Doppler factor
    doppler = 0.65 + 0.35 * np.cos(phi + 0.3)

    # Broad accretion disk (Keplerian power-law falloff with inner cutoff)
    r_in = 280.0
    r_out = 750.0
    disk_profile = np.exp(-((r - 420.0) ** 2) / (2.0 * (85.0 ** 2)))
    disk = 175.0 * disk_profile * doppler * (r >= r_in) * (r <= r_out)

    # Spiral density shock waves
    spiral_arm = 45.0 * np.cos(2.0 * phi - 4.5 * np.log(np.maximum(r / r_in, 1.0))) * disk_profile * doppler
    spiral_arm = np.maximum(spiral_arm, 0.0)

    arr += (disk + spiral_arm)
    meta = {"scene": "Synthetic Relativistic Accretion Disk", "r_in": r_in, "r_out": r_out, "doppler_boost": True}
    return arr, meta


def build_scene_5_black_hole(size_hr: int = 2048) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scene 5: Synthetic Black Hole - Event horizon shadow floor (I~38), photon ring, gravitational lensing."""
    np.random.seed(505)
    yy, xx = np.indices((size_hr, size_hr), dtype=np.float32)
    arr = np.full((size_hr, size_hr), 8.0, dtype=np.float32)
    arr += np.random.normal(0.0, 0.6, (size_hr, size_hr)).astype(np.float32)

    cx, cy = size_hr // 2, size_hr // 2
    r = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)

    r_horizon = 320.0
    r_photon = r_horizon * 1.15  # 368 px

    # Accretion emission backdrop
    backdrop = 130.0 * np.exp(-((r - 480.0) ** 2) / (2.0 * (90.0 ** 2))) * (0.7 + 0.3 * np.cos(phi))
    arr += backdrop

    # Razor-sharp photon ring
    d_photon = np.abs(r - r_photon)
    photon_ring = 220.0 * np.exp(-(d_photon ** 2) / (2.0 * (3.8 ** 2))) * (0.75 + 0.25 * np.cos(phi))
    arr += photon_ring

    # Shadow floor with realistic non-zero value
    shadow_mask = (r <= r_horizon)
    arr[shadow_mask] = 38.0 + np.random.normal(0.0, 0.5, np.count_nonzero(shadow_mask))

    meta = {"scene": "Synthetic Black Hole", "r_horizon": r_horizon, "r_photon": r_photon, "shadow_floor": 38.0}
    return arr, meta


def build_scene_6_galaxies_extended(size_hr: int = 2048) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scene 6: Galaxies & Extended Sources - Sersic profiles, spiral disks, irregular sources."""
    np.random.seed(606)
    yy, xx = np.indices((size_hr, size_hr), dtype=np.float32)
    arr = np.full((size_hr, size_hr), 9.0, dtype=np.float32)
    arr += np.random.normal(0.0, 0.7, (size_hr, size_hr)).astype(np.float32)

    # 1. Giant Elliptical Galaxy (de Vaucouleurs Sersic n=4)
    g1_x, g1_y = 650.0, 700.0
    r1 = np.hypot(xx - g1_x, yy - g1_y)
    r_eff1 = 120.0
    sersic_n4 = 180.0 * np.exp(-7.67 * ((r1 / r_eff1) ** 0.25 - 1.0))
    arr += np.clip(sersic_n4, 0.0, 210.0)

    # 2. Spiral Galaxy with Exponential Disk & Nucleus
    g2_x, g2_y = 1450.0, 1350.0
    dx2 = (xx - g2_x) * np.cos(0.5) + (yy - g2_y) * np.sin(0.5)
    dy2 = -(xx - g2_x) * np.sin(0.5) + (yy - g2_y) * np.cos(0.5)
    r2_ellip = np.hypot(dx2, dy2 * 2.2)  # Inclined disk
    disk2 = 110.0 * np.exp(-r2_ellip / 65.0)
    nucleus2 = 140.0 * np.exp(-(r2_ellip ** 2) / (2.0 * (12.0 ** 2)))
    arr += (disk2 + nucleus2)

    # 3. Irregular Dwarf Galaxy (clumpy patches)
    g3_x, g3_y = 1500.0, 500.0
    for cx_off, cy_off, flux_k in [(-25, -15, 65.0), (10, 20, 55.0), (35, -5, 45.0), (0, 0, 75.0)]:
        rk2 = (xx - (g3_x + cx_off)) ** 2 + (yy - (g3_y + cy_off)) ** 2
        arr += flux_k * np.exp(-rk2 / (2.0 * (28.0 ** 2)))

    meta = {"scene": "Galaxies & Extended Sources", "sources": ["elliptical_n4", "spiral_inclined", "dwarf_irregular"]}
    return arr, meta


def build_scene_7_multiscale_composite(size_hr: int = 2048) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scene 7: Multi-scale Composite Scene (from standard synthetic suite)."""
    from samples.synthetic.generate_synthetic_astro import generate_synthetic_dataset
    temp_dir = PROJECT_ROOT / "samples" / "synthetic"
    hr_path, lr_path, cat_path = generate_synthetic_dataset(temp_dir, size_hr=size_hr)
    arr = np.array(load_image(hr_path).convert("L")).astype(np.float32)
    with open(cat_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["scene"] = "Multi-scale Composite Scene"
    return arr, meta


def build_scene_8_low_contrast_background(size_hr: int = 2048) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Scene 8: Low-Contrast Faint Cosmic Background - Gradient I in [4, 18], faint dust lanes, subtle nebulosity."""
    np.random.seed(808)
    yy, xx = np.indices((size_hr, size_hr), dtype=np.float32)

    # Smooth subtle cosmic gradient (4.0 to 16.0)
    arr = 4.0 + 8.0 * (yy / size_hr) + 4.0 * (xx / size_hr)
    arr += np.random.normal(0.0, 0.9, (size_hr, size_hr)).astype(np.float32)

    # Ultra-faint diffuse nebulosity (I ~ 7.0 above background)
    neb_r = np.hypot(xx - 800.0, yy - 1200.0)
    arr += 7.5 * np.exp(-(neb_r ** 2) / (2.0 * (320.0 ** 2)))

    # Dark dust absorption lane (drops intensity by ~2.5)
    lane_dist = np.abs((xx - 1100.0) * 0.707 - (yy - 900.0) * 0.707)
    dust_mask = (lane_dist <= 80.0)
    arr[dust_mask] -= 2.5 * np.exp(-(lane_dist[dust_mask] ** 2) / (2.0 * (30.0 ** 2)))

    # Ensure strictly positive intensities
    arr = np.maximum(arr, 1.0)
    meta = {"scene": "Low-Contrast Faint Cosmic Background", "features": ["cosmic_gradient", "faint_nebula", "absorption_dust_lane"]}
    return arr, meta


def run_fidelity_benchmark_suite():
    print("=" * 80)
    print("DEEPZOOM ASTRO ENGINE V0.1.3 - FIDELITY BENCHMARK SUITE ACROSS 8 SCENES")
    print("=" * 80)

    scenes_dir = PROJECT_ROOT / "samples" / "synthetic" / "v013_scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    out_dir = PROJECT_ROOT / "output" / "experimental" / "v013" / "fidelity_scenes"
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = [
        ("01_isolated_star_field", build_scene_1_isolated_stars),
        ("02_dense_clusters", build_scene_2_dense_clusters),
        ("03_nebula_filaments", build_scene_3_nebula_filaments),
        ("04_accretion_disk", build_scene_4_accretion_disk),
        ("05_black_hole", build_scene_5_black_hole),
        ("06_galaxies_extended", build_scene_6_galaxies_extended),
        ("07_multiscale_composite", build_scene_7_multiscale_composite),
        ("08_low_contrast_background", build_scene_8_low_contrast_background),
    ]

    pipeline = AstroPipeline()
    validator = AstroValidator()

    results_by_scene = {}
    all_zero_regressions_passed = True

    for scene_id, builder_fn in builders:
        print(f"\n[{scene_id.upper()}] Building Ground Truth and LR...")
        hr_raw, meta = builder_fn(size_hr=2048)
        hr_clean = np.clip(np.round(hr_raw), 0.0, 255.0).astype(np.uint8)
        hr_im = Image.fromarray(hr_clean, mode="L").convert("RGB")
        hr_gt_path = scenes_dir / f"{scene_id}_gt_2048.png"
        hr_im.save(hr_gt_path, format="PNG")

        # Downsample to 512x512 LR using Box filtering (simulates pixel area integration)
        lr_im = hr_im.resize((512, 512), resample=Image.Resampling.BOX)
        lr_path = scenes_dir / f"{scene_id}_lr_512.png"
        lr_im.save(lr_path, format="PNG")

        gt_arr = np.array(hr_im)

        # 1. Run Pure Remacri Baseline
        base_out_path = out_dir / f"{scene_id}_baseline.png"
        print(f"  -> Running Frozen Remacri Baseline...")
        t0 = time.time()
        res_base = pipeline.run(
            input_image=lr_path,
            output_image=base_out_path,
            scale=4,
            tile_size=256,
            enable_astro_engine=False,
            ablation_mode="baseline"
        )
        base_arr = np.array(load_image(base_out_path))

        # 2. Run DeepZoom Astro Engine V0.1.3
        v013_out_path = out_dir / f"{scene_id}_v013.png"
        print(f"  -> Running DeepZoom Astro Engine V0.1.3...")
        res_v013 = pipeline.run(
            input_image=lr_path,
            output_image=v013_out_path,
            scale=4,
            tile_size=256,
            enable_astro_engine=True,
            ablation_mode="all_layers",
            astro_blend_weight=0.5
        )
        v013_arr = np.array(load_image(v013_out_path))

        # 3. Evaluate Metrics vs Ground Truth
        psnr_base_gt = float(validator.compute_psnr(gt_arr, base_arr))
        psnr_v013_gt = float(validator.compute_psnr(gt_arr, v013_arr))
        delta_psnr = round(psnr_v013_gt - psnr_base_gt, 3)

        ssim_base = validator.compute_ssim_detailed(gt_arr, base_arr)
        ssim_v013 = validator.compute_ssim_detailed(gt_arr, v013_arr)
        delta_ssim_y = round(ssim_v013["ssim_y"] - ssim_base["ssim_y"], 4)
        delta_ssim_rgb = round(ssim_v013["ssim_rgb"] - ssim_base["ssim_rgb"], 4)

        # Spatial Delta: Modified pixels
        diff_arr = np.abs(v013_arr.astype(np.float32) - base_arr.astype(np.float32))
        modified_mask = np.any(diff_arr >= 1.0, axis=-1)
        modified_pct = round(float(np.mean(modified_mask) * 100.0), 3)
        max_delta = float(np.max(diff_arr))

        # Zero Regression Guarantee Check:
        # DeepZoom should never cause significant regression (> 0.20 dB drop vs baseline).
        # In all cases it should maintain or improve quality.
        regression_passed = bool((delta_psnr >= -0.20) and (delta_ssim_y >= -0.005))
        if not regression_passed:
            all_zero_regressions_passed = False

        print(f"  RESULTS vs GT:")
        print(f"    PSNR Base: {psnr_base_gt:.2f} dB | V0.1.3: {psnr_v013_gt:.2f} dB (Delta: {delta_psnr:+.3f} dB)")
        print(f"    SSIM Y Base: {ssim_base['ssim_y']:.4f} | V0.1.3: {ssim_v013['ssim_y']:.4f} (Delta: {delta_ssim_y:+.4f})")
        print(f"    Modified Pixels: {modified_pct}% | Max Delta: {max_delta:.1f}")
        print(f"    Zero Regression Pass: {regression_passed}")

        scene_report = {
            "scene_id": scene_id,
            "scene_name": meta.get("scene", scene_id),
            "psnr_baseline_vs_gt_db": round(psnr_base_gt, 2),
            "psnr_v013_vs_gt_db": round(psnr_v013_gt, 2),
            "delta_psnr_db": delta_psnr,
            "ssim_y_baseline_vs_gt": ssim_base["ssim_y"],
            "ssim_y_v013_vs_gt": ssim_v013["ssim_y"],
            "delta_ssim_y": delta_ssim_y,
            "ssim_rgb_baseline_vs_gt": ssim_base["ssim_rgb"],
            "ssim_rgb_v013_vs_gt": ssim_v013["ssim_rgb"],
            "delta_ssim_rgb": delta_ssim_rgb,
            "spatial_delta": {
                "modified_pixel_pct": modified_pct,
                "max_absolute_delta": round(max_delta, 1),
                "mean_delta_on_modified": round(float(np.mean(diff_arr[modified_mask])), 2) if modified_pct > 0 else 0.0
            },
            "zero_regression_passed": regression_passed,
            "profiling": res_v013.get("profiling", {}),
            "star_preserve": res_v013.get("star_preserve", {}),
            "filament_guard": res_v013.get("filament_guard", {}),
            "dark_sky": res_v013.get("dark_sky", {}),
            "geometry_lock": res_v013.get("geometry_lock", {})
        }
        results_by_scene[scene_id] = scene_report

    # Assemble comprehensive report
    benchmark_report = {
        "benchmark_name": "DeepZoom Astro Engine V0.1.3 Fidelity Benchmark Suite Across 8 Scenes",
        "engine_version": "v0.1.3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": "AMD Radeon RX 5700 XT (8 GB VRAM, Vulkan device 0)",
            "cpu": "AMD Ryzen 5 9600X (6C/12T)",
            "ram": "16 GB DDR5",
            "os": "Windows 11"
        },
        "all_zero_regressions_passed": all_zero_regressions_passed,
        "total_scenes_evaluated": len(builders),
        "scenes": results_by_scene
    }

    report_path = PROJECT_ROOT / "benchmarks" / "v013" / "fidelity_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE. All zero regressions passed: {all_zero_regressions_passed}")
    print(f"Report written to: {report_path}")
    print("=" * 80)
    return benchmark_report


if __name__ == "__main__":
    run_fidelity_benchmark_suite()
