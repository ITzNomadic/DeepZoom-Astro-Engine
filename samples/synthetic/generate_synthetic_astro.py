"""
Expanded Synthetic Astronomical Dataset Generator for DeepZoom Studio V0.1.1.
Generates a scientifically rigorous benchmark scene with 4 distinct diagnostic groups:
  Group A: Isolated stars, close binary pairs, dense cluster, and faint threshold stars.
  Group B: Straight thin lines, curved filaments/arcs, and intersecting filaments.
  Group C: Smooth cosmic gradients, realistic noise floor, and faint diffuse emission nebula.
  Group D: Composite gravitational shadow with non-zero floor (I ~ 38), relativistic Doppler disk, and photon ring.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image

def generate_synthetic_dataset(output_dir: Path, size_hr: int = 2048, scale: int = 4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating expanded synthetic astronomical suite ({size_hr}x{size_hr})...")
    np.random.seed(1337)

    hr_arr = np.zeros((size_hr, size_hr), dtype=np.float32)
    yy, xx = np.indices((size_hr, size_hr), dtype=np.float32)

    # -------------------------------------------------------------
    # Group C: Cosmic Background, Smooth Gradients & Faint Nebula
    # -------------------------------------------------------------
    # 1. Smooth cosmic background gradient (from 3.0 to 14.0)
    grad_c = 3.0 + 8.0 * (yy / size_hr) + 3.0 * (xx / size_hr)
    # 2. Gaussian cosmic noise floor (std = 0.8)
    noise_c = np.random.normal(loc=0.0, scale=0.8, size=(size_hr, size_hr)).astype(np.float32)
    hr_arr += (grad_c + noise_c)

    # 3. Faint diffuse emission nebula in bottom-left region
    neb_cx, neb_cy = int(size_hr * 0.25), int(size_hr * 0.78)
    neb_r2 = ((xx - neb_cx) ** 2) / (2.0 * (180.0 ** 2)) + ((yy - neb_cy) ** 2) / (2.0 * (110.0 ** 2))
    nebula_patch = 22.0 * np.exp(-neb_r2) * (1.0 + 0.3 * np.sin(xx / 40.0) * np.cos(yy / 30.0))
    hr_arr += nebula_patch

    # -------------------------------------------------------------
    # Group D: Central Black Hole Shadow & Relativistic Accretion Disk
    # -------------------------------------------------------------
    cx, cy = size_hr // 2, size_hr // 2
    dist_center = np.hypot(xx - cx, yy - cy)
    phi = np.arctan2(yy - cy, xx - cx)

    # Accretion Disk Filaments with relativistic Doppler asymmetry (brighter on left)
    r_horizon = size_hr * 0.16  # radius = ~328 px
    doppler_factor = 0.65 + 0.35 * np.cos(phi + 0.4)

    # Photon ring (very sharp inner ridge)
    photon_dist = np.abs(dist_center - (r_horizon * 1.15))
    photon_ring = 210.0 * np.exp(-(photon_dist ** 2) / (2.0 * (4.5 ** 2))) * doppler_factor

    # Main relativistic accretion disk (broad ridge)
    disk_dist = np.abs(dist_center - (r_horizon * 1.45))
    main_disk = 150.0 * np.exp(-(disk_dist ** 2) / (2.0 * (16.0 ** 2))) * doppler_factor

    # Outer spiral arm filament
    spiral_phase = phi + 2.5 * np.log(np.maximum(dist_center / (r_horizon * 1.3), 1.0))
    spiral_dist = np.abs(dist_center - (r_horizon * 1.9 + 25.0 * np.sin(spiral_phase)))
    outer_spiral = 75.0 * np.exp(-(spiral_dist ** 2) / (2.0 * (8.0 ** 2)))

    hr_arr += (photon_ring + main_disk + outer_spiral)

    # Black hole shadow floor: realistic non-zero floor (I ~ 38.0)
    shadow_mask = dist_center <= r_horizon
    # Soft transition edge at horizon (2 pixels)
    transition = np.clip((dist_center - (r_horizon - 3.0)) / 4.0, 0.0, 1.0)
    shadow_floor_val = 38.0
    hr_arr = np.where(shadow_mask, shadow_floor_val + 0.5 * noise_c, hr_arr)

    # -------------------------------------------------------------
    # Group B: Filament Diagnostic Structures (Top-Left & Top-Right)
    # -------------------------------------------------------------
    # B1: Thin straight line at 35 degrees
    # Bounding box in top-left
    b1_x0, b1_y0 = size_hr * 0.12, size_hr * 0.22
    angle_b1 = np.radians(35.0)
    nx_b1 = -np.sin(angle_b1)
    ny_b1 = np.cos(angle_b1)
    d_line1 = np.abs((xx - b1_x0) * nx_b1 + (yy - b1_y0) * ny_b1)
    along_b1 = (xx - b1_x0) * np.cos(angle_b1) + (yy - b1_y0) * np.sin(angle_b1)
    line1_mask = (d_line1 <= 8.0) & (along_b1 >= 0) & (along_b1 <= 240.0)
    hr_arr[line1_mask] += 130.0 * np.exp(-(d_line1[line1_mask] ** 2) / (2.0 * (2.2 ** 2)))

    # B2: Intersecting cross filaments (at top-right)
    b2_cx, b2_cy = size_hr * 0.82, size_hr * 0.20
    d_cross1 = np.abs((xx - b2_cx) + (yy - b2_cy)) / np.sqrt(2.0)
    d_cross2 = np.abs((xx - b2_cx) - (yy - b2_cy)) / np.sqrt(2.0)
    cross_mask = (np.hypot(xx - b2_cx, yy - b2_cy) <= 120.0)
    cross1 = 110.0 * np.exp(-(d_cross1 ** 2) / (2.0 * (2.5 ** 2))) * cross_mask
    cross2 = 110.0 * np.exp(-(d_cross2 ** 2) / (2.0 * (2.5 ** 2))) * cross_mask
    hr_arr += (cross1 + cross2)

    # -------------------------------------------------------------
    # Group A: Stellar Diagnostic Sources
    # -------------------------------------------------------------
    stars_catalog = []
    star_id = 0

    def add_point_source(x: float, y: float, flux: float, sigma: float, label: str):
        nonlocal star_id
        r_box = int(np.ceil(sigma * 4.5))
        x0_s, x1_s = max(0, int(round(x)) - r_box), min(size_hr, int(round(x)) + r_box + 1)
        y0_s, y1_s = max(0, int(round(y)) - r_box), min(size_hr, int(round(y)) + r_box + 1)
        gyy, gxx = np.ogrid[y0_s:y1_s, x0_s:x1_s]
        psf = (flux / (2.0 * np.pi * (sigma ** 2))) * np.exp(-((gxx - x)**2 + (gyy - y)**2) / (2.0 * (sigma ** 2)))
        hr_arr[y0_s:y1_s, x0_s:x1_s] += psf
        stars_catalog.append({
            "id": star_id,
            "group": "Group_A",
            "label": label,
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "flux": round(float(flux), 1),
            "sigma": round(float(sigma), 2),
            "lr_projected_x": round(float(x / scale), 3),
            "lr_projected_y": round(float(y / scale), 3)
        })
        star_id += 1

    # A1: Isolated stars with diverse flux levels
    isolated_coords = [
        (size_hr * 0.15, size_hr * 0.45, 8000.0, 1.8, "isolated_bright"),
        (size_hr * 0.18, size_hr * 0.60, 4000.0, 1.6, "isolated_medium"),
        (size_hr * 0.85, size_hr * 0.48, 12000.0, 2.0, "isolated_very_bright"),
        (size_hr * 0.82, size_hr * 0.62, 5500.0, 1.7, "isolated_medium_2"),
        (size_hr * 0.30, size_hr * 0.15, 1200.0, 1.4, "isolated_faint")
    ]
    for x, y, f, s, lbl in isolated_coords:
        add_point_source(x, y, f, s, lbl)

    # A2: Close binary pairs (separation = 12 px in HR -> 3 px in LR)
    # Testing diffraction resolving capability without merging
    bin1_x, bin1_y = size_hr * 0.72, size_hr * 0.80
    add_point_source(bin1_x - 6.0, bin1_y, 5000.0, 1.5, "binary_pair_1a")
    add_point_source(bin1_x + 6.0, bin1_y, 4500.0, 1.5, "binary_pair_1b")

    bin2_x, bin2_y = size_hr * 0.68, size_hr * 0.25
    add_point_source(bin2_x, bin2_y - 7.0, 3500.0, 1.4, "binary_pair_2a")
    add_point_source(bin2_x, bin2_y + 7.0, 3200.0, 1.4, "binary_pair_2b")

    # A3: Dense globular cluster (15 stars clustered in bottom-right)
    clust_cx, clust_cy = size_hr * 0.82, size_hr * 0.82
    for k in range(12):
        angle = k * (2.0 * np.pi / 12.0)
        dist_k = 18.0 + (k % 3) * 14.0
        c_x = clust_cx + dist_k * np.cos(angle)
        c_y = clust_cy + dist_k * np.sin(angle)
        f_k = 2500.0 + (k % 4) * 800.0
        add_point_source(c_x, c_y, f_k, 1.4, f"cluster_star_{k+1}")

    # Clip dynamic range to [0, 255]
    hr_clean = np.clip(np.round(hr_arr), 0.0, 255.0).astype(np.uint8)
    hr_im = Image.fromarray(hr_clean, mode="L").convert("RGB")
    hr_path = output_dir / "synthetic_astro_ground_truth_2048.png"
    hr_im.save(hr_path, format="PNG")

    # Downsample to LR (512x512) using high-quality Box/Area resampling
    size_lr = size_hr // scale
    lr_im = hr_im.resize((size_lr, size_lr), resample=Image.Resampling.BOX)
    lr_path = output_dir / "synthetic_astro_input_512.png"
    lr_im.save(lr_path, format="PNG")

    # Complete catalog metadata
    catalog = {
        "dataset_name": "DeepZoom_Synthetic_Astro_Suite_V0.1.1",
        "description": "Multi-group astronomical benchmark suite: Group A (stars/binaries/cluster), Group B (filaments/crossings), Group C (cosmic background/nebula), Group D (black hole shadow/disk)",
        "scale_factor": scale,
        "hr_dimensions": [size_hr, size_hr],
        "lr_dimensions": [size_lr, size_lr],
        "event_horizon": {
            "center": [cx, cy],
            "radius_px": round(r_horizon, 2),
            "shadow_floor_intensity": shadow_floor_val
        },
        "groups": {
            "Group_A": {
                "name": "Stellar Point Sources",
                "total_stars": len(stars_catalog),
                "binary_pairs_count": 2,
                "cluster_stars_count": 12,
                "catalog": stars_catalog
            },
            "Group_B": {
                "name": "Filamentary Structures",
                "straight_filament": {"origin": [b1_x0, b1_y0], "length_px": 240, "angle_deg": 35.0},
                "intersecting_cross": {"center": [b2_cx, b2_cy], "radius_px": 120}
            },
            "Group_C": {
                "name": "Background & Diffuse Nebulosity",
                "emission_nebula": {"center": [neb_cx, neb_cy], "peak_intensity": 22.0},
                "cosmic_noise_sigma": 0.8
            },
            "Group_D": {
                "name": "Relativistic Gravitational Cavity",
                "event_horizon_radius": round(r_horizon, 2),
                "photon_ring_radius": round(r_horizon * 1.15, 2),
                "shadow_floor_intensity": shadow_floor_val
            }
        }
    }

    catalog_path = output_dir / "synthetic_ground_truth_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    print(f"[OK] Synthetic Suite V0.1.1 generated:")
    print(f"     Ground Truth HR (2048x2048): {hr_path}")
    print(f"     Input LR (512x512):          {lr_path}")
    print(f"     Catalog metadata:            {catalog_path}")
    return hr_path, lr_path, catalog_path

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    generate_synthetic_dataset(out_dir)
