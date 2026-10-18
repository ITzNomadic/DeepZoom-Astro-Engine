"""
Execution Script for DeepZoom Astro Engine V0.1.2 - Comparative Visual Matrix & Ablation Suite.
Generates:
1. All 5 pipeline modes on the real astronomical test image (2728x4096):
   - Baseline (Pure Remacri)
   - StarPreserve Only
   - FilamentGuard Only
   - DarkSky Only
   - All Layers V0.1.2 Full
2. The 8 mandatory 1:1 strategic crops (256x256 px):
   - Crop 1: Event Horizon
   - Crop 2: Accretion Disk Edge
   - Crop 3: Fine Filaments
   - Crop 4: Isolated Point Source Star
   - Crop 5: Crowded Star Cluster
   - Crop 6: Small Galaxy
   - Crop 7: Faint Nebulosity
   - Crop 8: Deep Cosmic Background
3. Side-by-side comparative matrices: [Remacri | V0.1.2 | Diff x15].
4. Amplified difference maps (x15 gain).
5. Visual manifest.json with coordinates, dimensions, parameters, and SHA-256 hashes.
6. Benchmarks: benchmarks/v012/benchmark_report.json and ablation_report.json.
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepzoom.core.config import DeepZoomConfig
from deepzoom.astro.star_preserve import StarPreserve
from deepzoom.astro.filament_guard import FilamentGuard
from deepzoom.astro.dark_sky import DarkSky
from deepzoom.astro.geometry_lock import GeometryLock
from deepzoom.fusion.adaptive_fuser import AdaptiveFuser
from deepzoom.validation.astro_validator import AstroValidator
from deepzoom.utils.profiler import AstroProfiler


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def create_side_by_side_panel(
    remacri_crop: np.ndarray,
    v012_crop: np.ndarray,
    diff_crop: np.ndarray,
    title: str
) -> np.ndarray:
    """
    Combines 3 panels: [Pure Remacri | DeepZoom V0.1.2 | Diff x15]
    with clear header labels.
    """
    ch, cw, _ = remacri_crop.shape
    header_h = 36
    panel_w = cw * 3 + 4  # 2 px separators
    panel_h = ch + header_h

    canvas = np.full((panel_h, panel_w, 3), 18, dtype=np.uint8)

    # Place crops
    canvas[header_h:, :cw, :] = remacri_crop
    # 2 px divider
    canvas[header_h:, cw:cw+2, :] = 80
    canvas[header_h:, cw+2:2*cw+2, :] = v012_crop
    # 2 px divider
    canvas[header_h:, 2*cw+2:2*cw+4, :] = 80
    canvas[header_h:, 2*cw+4:, :] = diff_crop

    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)

    # Header text
    draw.text((10, 8), f"Remacri Pure Baseline — {title}", fill=(200, 200, 200))
    draw.text((cw + 12, 8), "DeepZoom Astro Engine V0.1.2", fill=(100, 220, 255))
    draw.text((2 * cw + 14, 8), "Difference Map (15x Gain)", fill=(255, 180, 80))

    return np.array(im)


def run_comparative_matrix():
    print("================================================================================")
    print("DEEPZOOM ASTRO ENGINE V0.1.2 - COMPARATIVE VISUAL MATRIX & ABLATION")
    print("================================================================================")

    in_lr_path = PROJECT_ROOT / "samples" / "astronomical" / "deepzoom_astronomy_test.png"
    baseline_hr_path = PROJECT_ROOT / "output" / "experimental" / "v011" / "astronomy_test_4k_v011_baseline.png"

    out_v012_dir = PROJECT_ROOT / "output" / "experimental" / "v012"
    out_v012_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_v012_dir / "crops" / "matrix"
    crops_dir.mkdir(parents=True, exist_ok=True)
    diff_dir = out_v012_dir / "diff_maps"
    diff_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = PROJECT_ROOT / "benchmarks" / "v012"
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Load LR input and baseline HR
    print(f"Loading LR input: {in_lr_path}")
    im_lr = Image.open(in_lr_path)
    in_gray = np.array(im_lr.convert("L"))

    print(f"Loading 4K Baseline HR: {baseline_hr_path}")
    im_base = Image.open(baseline_hr_path)
    base_rgb = np.array(im_base)
    base_gray = np.array(im_base.convert("L"))
    h_hr, w_hr, _ = base_rgb.shape
    eff_scale = float(w_hr / im_lr.size[0])
    print(f"  Dimensions: {w_hr}x{h_hr}, Effective Scale: {eff_scale}x")

    # Initialize V0.1.2 Modules
    sp = StarPreserve()
    fg = FilamentGuard(ridge_threshold=0.08, coherence_threshold=0.30, min_filament_length=15)
    ds = DarkSky()
    gl = GeometryLock()
    fuser = AdaptiveFuser(default_weight=0.5)
    validator = AstroValidator()
    profiler = AstroProfiler()

    # Step 1: Precompute Astronomical Layers (Strict Single-Weight Contract)
    print("\n--- Running DeepZoom Astro Engine V0.1.2 Analysis & Reconstructions ---")
    
    with profiler.measure("star_preserve_analysis"):
        in_stars = sp.analyze_stars(in_gray)
        out_stars = sp.analyze_stars(base_gray)
        star_catalog = sp.compare_catalogs(in_stars, out_stars, scale_factor=eff_scale)
        star_recon_rgb, star_mask, star_meta = sp.reconstruct_point_sources(
            baseline_rgb=base_rgb,
            input_stars=in_stars,
            output_stars=out_stars,
            scale_factor=eff_scale,
            blend_weight=1.0
        )
    print(f"  StarPreserve: {star_catalog['matched_stars_count']} matched, {star_meta['reconstructed_stars']} reconstructed")

    with profiler.measure("filament_guard_analysis"):
        fil_analysis = fg.analyze_filaments(base_gray)
        fil_repaired_rgb, fil_mask, fil_meta = fg.repair_filaments(
            baseline_rgb=base_rgb,
            analysis=fil_analysis,
            pure_repair=True
        )
    print(f"  FilamentGuard: {fil_analysis['filament_coverage_pct']}% coverage, {fil_meta['repaired_pixels']} repaired pixels")

    with profiler.measure("dark_sky_analysis"):
        dark_analysis = ds.analyze_background(base_gray)
        dark_rgb, dark_mask, dark_meta = ds.stabilize_vacuum(
            image_rgb=base_rgb,
            analysis=dark_analysis,
            pure_repair=True
        )
    print(f"  DarkSky: {dark_analysis['flat_vacuum_coverage_pct']}% flat vacuum, {dark_meta['stabilized_pixels']} stabilized")

    with profiler.measure("geometry_lock_verification"):
        geo_res = gl.evaluate_all(in_gray, base_gray, expected_scale=eff_scale)
    print(f"  GeometryLock: General Lock Passed = {geo_res['geometric_lock_passed']}")

    # Step 2: Generate All 5 Pipeline Modes & Ablation
    ablation_modes = [
        "baseline",
        "star_preserve_only",
        "filament_guard_only",
        "dark_sky_only",
        "all_layers"
    ]
    ablation_results = {}
    fused_images = {}

    print("\n--- Generating All 5 Pipeline Modes ---")
    for mode in ablation_modes:
        with profiler.measure(f"fusion_mode_{mode}"):
            fused, f_stats = fuser.fuse(
                baseline_rgb=base_rgb,
                mode=mode,
                star_reconstructed_rgb=star_recon_rgb,
                star_mask=star_mask,
                filament_repaired_rgb=fil_repaired_rgb,
                filament_mask=fil_mask,
                dark_sky_rgb=dark_rgb,
                dark_sky_mask=dark_mask,
                weight=0.5
            )
            out_mode_path = out_v012_dir / f"astronomy_test_4k_v012_{mode}.png"
            Image.fromarray(fused).save(out_mode_path, format="PNG")
            fused_images[mode] = fused

            val_metrics = validator.evaluate_all(base_rgb, fused)
            ablation_results[mode] = {
                "file_path": str(out_mode_path),
                "sha256": sha256_file(out_mode_path),
                "psnr_db": val_metrics["psnr_db"],
                "ssim": val_metrics["ssim"],
                "fusion_stats": f_stats,
                "validation_delta": val_metrics["spatial_delta"]
            }
            print(f"  Mode [{mode}]: PSNR={val_metrics['psnr_db']} dB, SSIM={val_metrics['ssim']}, Mod={f_stats['effectively_modified_pct']}%")

    # Step 3: Full-Master Difference Maps
    print("\n--- Generating Full Difference Maps ---")
    master_diff_maps = fuser.generate_difference_maps(
        baseline_rgb=base_rgb,
        fused_rgb=fused_images["all_layers"],
        star_reconstructed_rgb=star_recon_rgb,
        filament_repaired_rgb=fil_repaired_rgb,
        dark_sky_rgb=dark_rgb,
        amplification_factor=15.0
    )
    saved_diff_paths = fuser.save_difference_maps(master_diff_maps, diff_dir, prefix="master_v012")
    for dname, dpath in saved_diff_paths.items():
        print(f"  {dname}: {dpath}")

    # Step 4: Extract 8 Mandatory Strategic 1:1 Crops
    strategic_crops_def = [
        {
            "id": 1,
            "name": "crop1_event_horizon",
            "title": "Event Horizon (Shadow Inner Boundary)",
            "y": 1760,
            "x": 1308,
            "size": 256,
            "description": "Inner shadow boundary and strong gravitational deflection zone."
        },
        {
            "id": 2,
            "name": "crop2_accretion_disk",
            "title": "Accretion Disk Edge & Photon Ring",
            "y": 1560,
            "x": 1308,
            "size": 256,
            "description": "Bright relativistic photon ring and upper accretion disk boundary."
        },
        {
            "id": 3,
            "name": "crop3_fine_filaments",
            "title": "Fine Plasma Filaments",
            "y": 1408,
            "x": 1664,
            "size": 256,
            "description": "High-coherence plasma streams and directional relativistic filaments."
        },
        {
            "id": 4,
            "name": "crop4_isolated_star",
            "title": "Isolated Point Source Star",
            "y": 3572,
            "x": 0,
            "size": 256,
            "description": "Isolated stellar point source verifying PSF symmetry, flux conservation and anti-ringing."
        },
        {
            "id": 5,
            "name": "crop5_crowded_cluster",
            "title": "Crowded Star Cluster",
            "y": 3309,
            "x": 2149,
            "size": 256,
            "description": "High-density stellar field verifying distinct source resolution without merging."
        },
        {
            "id": 6,
            "name": "crop6_small_galaxy",
            "title": "Small Extended Galaxy",
            "y": 2697,
            "x": 2332,
            "size": 256,
            "description": "Extended non-point source validating preservation of diffuse galactic morphology."
        },
        {
            "id": 7,
            "name": "crop7_faint_nebulosity",
            "title": "Faint Nebulosity & Dust",
            "y": 2304,
            "x": 2432,
            "size": 256,
            "description": "Low-surface-brightness reflection gas and dust clouds preserved without black clipping."
        },
        {
            "id": 8,
            "name": "crop8_deep_cosmic_background",
            "title": "Deep Cosmic Background",
            "y": 64,
            "x": 1152,
            "size": 256,
            "description": "Cosmic vacuum floor testing posterization suppression and smooth gradient preservation."
        }
    ]

    print("\n--- Extracting and Generating 8 Strategic 1:1 Crops ---")
    v012_full_master = fused_images["all_layers"]
    crop_manifest = []

    for cdef in strategic_crops_def:
        cid = cdef["id"]
        cname = cdef["name"]
        ctitle = cdef["title"]
        cy, cx, csize = cdef["y"], cdef["x"], cdef["size"]

        # Ensure within bounds
        y0 = max(0, min(cy, h_hr - csize))
        y1 = y0 + csize
        x0 = max(0, min(cx, w_hr - csize))
        x1 = x0 + csize

        # Extract 1:1 crops
        crop_base = base_rgb[y0:y1, x0:x1, :]
        crop_v012 = v012_full_master[y0:y1, x0:x1, :]

        # Difference map for crop (15x gain)
        diff_raw = np.abs(crop_v012.astype(np.float32) - crop_base.astype(np.float32))
        crop_diff_15x = np.clip(np.round(diff_raw * 15.0), 0.0, 255.0).astype(np.uint8)

        # Save individual crop files
        path_base_crop = crops_dir / f"{cname}_01_pure_remacri.png"
        path_v012_crop = crops_dir / f"{cname}_02_deepzoom_v012.png"
        path_diff_crop = crops_dir / f"{cname}_03_diff_amplified_15x.png"
        path_panel = crops_dir / f"{cname}_04_side_by_side_panel.png"

        Image.fromarray(crop_base).save(path_base_crop)
        Image.fromarray(crop_v012).save(path_v012_crop)
        Image.fromarray(crop_diff_15x).save(path_diff_crop)

        # Build side-by-side panel
        panel = create_side_by_side_panel(crop_base, crop_v012, crop_diff_15x, ctitle)
        Image.fromarray(panel).save(path_panel)

        # Compute crop-specific metrics
        crop_psnr = validator.compute_psnr(crop_base, crop_v012)
        crop_ssim = validator.compute_ssim(crop_base, crop_v012)
        max_diff = float(np.max(diff_raw))
        mean_diff = float(np.mean(diff_raw))
        mod_pixels = int(np.sum(np.max(diff_raw, axis=-1) >= 0.5))
        mod_pct = float(mod_pixels / (csize * csize) * 100.0)

        entry = {
            "crop_id": cid,
            "crop_name": cname,
            "crop_title": ctitle,
            "description": cdef["description"],
            "coordinates": {
                "x0": x0, "x1": x1,
                "y0": y0, "y1": y1,
                "width": csize, "height": csize
            },
            "metrics": {
                "psnr_db": crop_psnr,
                "ssim": crop_ssim,
                "mean_absolute_diff": round(mean_diff, 4),
                "max_absolute_diff": round(max_diff, 2),
                "modified_pixels": mod_pixels,
                "modified_pct": round(mod_pct, 2)
            },
            "files": {
                "pure_remacri": str(path_base_crop.name),
                "deepzoom_v012": str(path_v012_crop.name),
                "diff_amplified_15x": str(path_diff_crop.name),
                "side_by_side_panel": str(path_panel.name)
            },
            "hashes": {
                "pure_remacri_sha256": sha256_file(path_base_crop),
                "deepzoom_v012_sha256": sha256_file(path_v012_crop),
                "diff_sha256": sha256_file(path_diff_crop),
                "panel_sha256": sha256_file(path_panel)
            }
        }
        crop_manifest.append(entry)
        print(f"  Crop {cid} [{cname}]: PSNR={crop_psnr} dB, SSIM={crop_ssim}, Mod={mod_pct}% ({mod_pixels} px)")

    # Save manifest.json
    manifest_path = crops_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "0.1.2",
            "release": "DeepZoom Astro Engine V0.1.2 Precision & Fidelity",
            "scale": 4,
            "base_model": "remacri-4x",
            "blend_weight": 0.5,
            "single_weight_contract": True,
            "total_crops": len(crop_manifest),
            "crops": crop_manifest
        }, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")

    # Step 5: Save Ablation & Benchmark Reports
    ablation_report_path = bench_dir / "ablation_report.json"
    with open(ablation_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "0.1.2",
            "description": "DeepZoom Astro Engine V0.1.2 Layer Ablation Benchmark on 4K Master",
            "master_dimensions": [w_hr, h_hr],
            "effective_scale": eff_scale,
            "modes": ablation_results
        }, f, indent=2)
    print(f"Ablation report saved to: {ablation_report_path}")

    benchmark_report_path = bench_dir / "benchmark_report.json"
    with open(benchmark_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "0.1.2",
            "release": "DeepZoom Astro Engine V0.1.2",
            "profiling_summary": profiler.get_summary(),
            "star_preserve": {
                "matched_sources": star_catalog["matched_stars_count"],
                "point_precision": star_catalog["point_source_precision"],
                "reconstructed_stars": star_meta["reconstructed_stars"],
                "mean_centroid_drift_px": star_catalog["mean_centroid_drift_px"]
            },
            "filament_guard": {
                "coverage_pct": fil_analysis["filament_coverage_pct"],
                "mean_coherence": fil_analysis["mean_filament_coherence"],
                "repaired_pixels": fil_meta["repaired_pixels"]
            },
            "dark_sky": {
                "flat_vacuum_coverage_pct": dark_analysis["flat_vacuum_coverage_pct"],
                "faint_structure_coverage_pct": dark_analysis["faint_structure_coverage_pct"],
                "stabilized_pixels": dark_meta["stabilized_pixels"]
            },
            "geometry_lock": {
                "general_lock_passed": geo_res["geometric_lock_passed"],
                "shadow_detected": geo_res["semantic_shadow_analysis"]["shadow_tracking_verified"]
            },
            "fusion_stats": ablation_results["all_layers"]["fusion_stats"],
            "validation_metrics": ablation_results["all_layers"]["validation_delta"]
        }, f, indent=2)
    print(f"Overall benchmark report saved to: {benchmark_report_path}")

    print("\n>>> ALL V0.1.2 COMPARATIVE MATRIX & ABLATION RUNS COMPLETED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    run_comparative_matrix()
