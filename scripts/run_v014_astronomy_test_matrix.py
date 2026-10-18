"""
Execution Script for DeepZoom Astro Engine V0.1.4 - 4K Master Matrix & 10 Mandatory Variants.
Evaluates on samples/astronomical/deepzoom_astronomy_test.png (2728x4096, 11.17 MP):
  1. Remacri Baseline (Pure frozen baseline)
  2. UltraSharp-4x (Pure neural model)
  3. RealESRGAN-x4plus (Pure neural model)
  4. High-Fidelity-4x (Pure neural model)
  5. Remacri + BackProjection (IBP consistency)
  6. Remacri + Star Reconstruction (PSF & binary separation)
  7. Remacri + Filament Reconstruction (Structure tensor & geometry)
  8. MultiScale Fusion (Laplacian pyramid multiscale)
  9. DeepZoom Fidelity (Complete scientific mode)
  10. DeepZoom Cinematic (Complete perceptual mode)

Extracts strategic 1:1 native-resolution crops, builds comparative panels,
generates selection maps, difference maps, and visual manifest.json.
"""

import sys
import json
import hashlib
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from deepzoom.pipeline import AstroPipeline
from deepzoom.validation.astro_validator import AstroValidator
from deepzoom.utils.image_io import load_image, get_image_info


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def create_quad_panel(
    remacri_crop: np.ndarray,
    fidelity_crop: np.ndarray,
    cinematic_crop: np.ndarray,
    diff_crop: np.ndarray,
    title: str
) -> np.ndarray:
    """
    Combines 4 panels: [Remacri Pure | DeepZoom Fidelity | DeepZoom Cinematic | Diff 15x]
    """
    ch, cw, _ = remacri_crop.shape
    header_h = 36
    panel_w = cw * 4 + 6
    panel_h = ch + header_h

    canvas = np.full((panel_h, panel_w, 3), 16, dtype=np.uint8)

    # Place crops with 2px dividers
    canvas[header_h:, :cw, :] = remacri_crop
    canvas[header_h:, cw:cw+2, :] = 70
    canvas[header_h:, cw+2:2*cw+2, :] = fidelity_crop
    canvas[header_h:, 2*cw+2:2*cw+4, :] = 70
    canvas[header_h:, 2*cw+4:3*cw+4, :] = cinematic_crop
    canvas[header_h:, 3*cw+4:3*cw+6, :] = 70
    canvas[header_h:, 3*cw+6:, :] = diff_crop

    im = Image.fromarray(canvas)
    draw = ImageDraw.Draw(im)

    draw.text((10, 8), f"Remacri Pure — {title}", fill=(190, 190, 190))
    draw.text((cw + 12, 8), "DeepZoom Fidelity V0.1.4", fill=(80, 210, 255))
    draw.text((2 * cw + 14, 8), "DeepZoom Cinematic V0.1.4", fill=(255, 170, 70))
    draw.text((3 * cw + 16, 8), "Difference Map (15x Gain)", fill=(240, 100, 100))

    return np.array(im)


def run_v014_astronomy_matrix():
    print("=" * 80)
    print("DEEPZOOM ASTRO ENGINE V0.1.4 - 4K ASTRONOMICAL MASTER MATRIX & 10 VARIANTS")
    print("=" * 80)

    in_lr_path = PROJECT_ROOT / "samples" / "astronomical" / "deepzoom_astronomy_test.png"
    out_dir = PROJECT_ROOT / "output" / "experimental" / "v014"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops" / "matrix"
    crops_dir.mkdir(parents=True, exist_ok=True)
    diff_dir = out_dir / "diff_maps"
    diff_dir.mkdir(parents=True, exist_ok=True)
    bench_dir = PROJECT_ROOT / "benchmarks" / "v014"
    bench_dir.mkdir(parents=True, exist_ok=True)

    pipeline = AstroPipeline()
    validator = AstroValidator()

    # The 10 Mandatory Pipeline Modes
    mandatory_variants = [
        ("01_remacri_baseline", "baseline", False),
        ("02_ultrasharp", "ultrasharp", False),
        ("03_realesrgan", "realesrgan", False),
        ("04_high_fidelity", "high_fidelity", False),
        ("05_remacri_plus_ibp", "remacri_plus_ibp", True),
        ("06_remacri_plus_star", "remacri_plus_star", True),
        ("07_remacri_plus_filament", "remacri_plus_filament", True),
        ("08_multiscale_fusion", "multiscale_fusion", True),
        ("09_deepzoom_fidelity", "deepzoom_fidelity", True),
        ("10_deepzoom_cinematic", "deepzoom_cinematic", True),
    ]

    master_results = {}
    master_arrays = {}

    print("\n--- Generating All 10 Master Variants on 4K Astronomy Test Image ---")
    for var_id, mode_name, enable_engine in mandatory_variants:
        out_var_path = out_dir / f"astronomy_test_4k_v014_{var_id}.png"
        print(f"  >> Running Variant [{var_id}]: mode='{mode_name}'...")
        t0 = time.time()
        res = pipeline.run(
            input_image=in_lr_path,
            output_image=out_var_path,
            scale=4,
            tile_size=400,
            enable_astro_engine=enable_engine,
            ablation_mode=mode_name
        )
        elapsed = time.time() - t0

        arr = np.array(load_image(out_var_path))
        master_arrays[var_id] = arr
        sha = sha256_file(out_var_path)

        master_results[var_id] = {
            "variant_id": var_id,
            "mode": mode_name,
            "astro_engine_enabled": enable_engine,
            "file_path": str(out_var_path.relative_to(PROJECT_ROOT)),
            "sha256": sha,
            "execution_seconds": round(elapsed, 2),
            "dimensions": [arr.shape[1], arr.shape[0]],
            "meta_artifacts": res.get("meta_artifacts", {})
        }
        print(f"     Finished in {elapsed:.2f}s | SHA-256: {sha[:16]}...")

    # Establish Remacri Baseline array for comparisons
    remacri_arr = master_arrays["01_remacri_baseline"]
    fidelity_arr = master_arrays["09_deepzoom_fidelity"]
    cinematic_arr = master_arrays["10_deepzoom_cinematic"]

    # Generate Full-Master Difference Maps
    print("\n--- Generating 4K Master Difference Maps ---")
    diff_fid_raw = np.abs(fidelity_arr.astype(np.float32) - remacri_arr.astype(np.float32))
    diff_fid_15x = np.clip(np.round(diff_fid_raw * 15.0), 0.0, 255.0).astype(np.uint8)
    diff_fid_path = diff_dir / "master_v014_diff_fidelity_15x.png"
    Image.fromarray(diff_fid_15x).save(diff_fid_path)

    diff_cin_raw = np.abs(cinematic_arr.astype(np.float32) - remacri_arr.astype(np.float32))
    diff_cin_15x = np.clip(np.round(diff_cin_raw * 15.0), 0.0, 255.0).astype(np.uint8)
    diff_cin_path = diff_dir / "master_v014_diff_cinematic_15x.png"
    Image.fromarray(diff_cin_15x).save(diff_cin_path)

    # -------------------------------------------------------------------------
    # Extract Priority 1:1 Crops and Build Comparative Quad Panels
    # -------------------------------------------------------------------------
    print("\n--- Extracting 1:1 Strategic Crops and Building Panels ---")
    strategic_crops = [
        {"id": 1, "name": "crop1_event_horizon", "title": "Borde del Agujero Negro y Horizonte", "y": 1760, "x": 1308, "size": 256},
        {"id": 2, "name": "crop2_accretion_disk", "title": "Anillo Fotónico y Filamentos del Disco", "y": 1560, "x": 1308, "size": 256},
        {"id": 3, "name": "crop3_fine_filaments", "title": "Filamentos Finos de Plasma", "y": 1408, "x": 1664, "size": 256},
        {"id": 4, "name": "crop4_isolated_star", "title": "Estrella Pequeña Puntual Aislada", "y": 3572, "x": 0, "size": 256},
        {"id": 5, "name": "crop5_crowded_cluster", "title": "Cúmulo Estelar Denso", "y": 3309, "x": 2149, "size": 256},
        {"id": 6, "name": "crop6_small_galaxy", "title": "Galaxia Extendida Lejana", "y": 2697, "x": 2332, "size": 256},
        {"id": 7, "name": "crop7_faint_nebulosity", "title": "Nebulosidad Tenue y Polvo", "y": 2304, "x": 2432, "size": 256},
        {"id": 8, "name": "crop8_deep_cosmic_background", "title": "Fondo Oscuro con Estructuras Débiles", "y": 64, "x": 1152, "size": 256},
        {"id": 9, "name": "crop9_binary_pair", "title": "Pareja Estelar Doble", "y": 3450, "x": 2200, "size": 256}
    ]

    h_hr, w_hr, _ = remacri_arr.shape
    crop_manifest = []

    for cdef in strategic_crops:
        cid = cdef["id"]
        cname = cdef["name"]
        ctitle = cdef["title"]
        cy, cx, csize = cdef["y"], cdef["x"], cdef["size"]

        y0 = max(0, min(cy, h_hr - csize))
        y1 = y0 + csize
        x0 = max(0, min(cx, w_hr - csize))
        x1 = x0 + csize

        # Extract crops
        c_rem = remacri_arr[y0:y1, x0:x1, :]
        c_fid = fidelity_arr[y0:y1, x0:x1, :]
        c_cin = cinematic_arr[y0:y1, x0:x1, :]

        # Difference map for crop (15x)
        diff_raw = np.abs(c_fid.astype(np.float32) - c_rem.astype(np.float32))
        c_diff_15x = np.clip(np.round(diff_raw * 15.0), 0.0, 255.0).astype(np.uint8)

        # Save individual crop files
        p_rem = crops_dir / f"{cname}_01_pure_remacri.png"
        p_fid = crops_dir / f"{cname}_02_deepzoom_fidelity.png"
        p_cin = crops_dir / f"{cname}_03_deepzoom_cinematic.png"
        p_diff = crops_dir / f"{cname}_04_diff_15x.png"
        p_panel = crops_dir / f"{cname}_05_quad_panel.png"

        Image.fromarray(c_rem).save(p_rem)
        Image.fromarray(c_fid).save(p_fid)
        Image.fromarray(c_cin).save(p_cin)
        Image.fromarray(c_diff_15x).save(p_diff)

        # Build quad panel
        panel = create_quad_panel(c_rem, c_fid, c_cin, c_diff_15x, ctitle)
        Image.fromarray(panel).save(p_panel)

        val_fid = validator.evaluate_all(c_rem, c_fid)
        val_cin = validator.evaluate_all(c_rem, c_cin)

        entry = {
            "crop_id": cid,
            "crop_name": cname,
            "crop_title": ctitle,
            "coordinates": {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "width": csize, "height": csize},
            "metrics_vs_remacri": {
                "fidelity": {"psnr_db": val_fid["psnr_db"], "ssim": val_fid["ssim"], "mod_pct": val_fid["spatial_delta"]["modified_pixel_pct"]},
                "cinematic": {"psnr_db": val_cin["psnr_db"], "ssim": val_cin["ssim"], "mod_pct": val_cin["spatial_delta"]["modified_pixel_pct"]}
            },
            "files": {
                "remacri": str(p_rem.relative_to(PROJECT_ROOT)),
                "fidelity": str(p_fid.relative_to(PROJECT_ROOT)),
                "cinematic": str(p_cin.relative_to(PROJECT_ROOT)),
                "diff_15x": str(p_diff.relative_to(PROJECT_ROOT)),
                "quad_panel": str(p_panel.relative_to(PROJECT_ROOT))
            },
            "sha256": {
                "remacri": sha256_file(p_rem),
                "fidelity": sha256_file(p_fid),
                "cinematic": sha256_file(p_cin)
            }
        }
        crop_manifest.append(entry)
        print(f"  [Crop {cid}] {cname}: Fidelity Mod={val_fid['spatial_delta']['modified_pixel_pct']}%, Cinematic Mod={val_cin['spatial_delta']['modified_pixel_pct']}%")

    # Save Manifest
    manifest_doc = {
        "engine_version": "DeepZoom Astro Engine V0.1.4",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu": "AMD Radeon RX 5700 XT (8 GB VRAM)",
            "cpu": "AMD Ryzen 5 9600X (6C/12T)",
            "os": "Windows 11"
        },
        "master_variants": master_results,
        "crops": crop_manifest
    }

    manifest_path = crops_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)

    bench_path = bench_dir / "astronomy_4k_report.json"
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)

    print(f"\nManifest successfully written: {manifest_path}")
    print("=" * 80)
    return manifest_doc


if __name__ == "__main__":
    run_v014_astronomy_matrix()
