"""
Generates web-optimized showcase previews for GitHub README rendering.
Links to the full-resolution uncompressed 12K, 8K, and 4K masters.
"""

from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_previews():
    showcase_dir = PROJECT_ROOT / "docs" / "showcase"
    showcase_dir.mkdir(parents=True, exist_ok=True)

    # 1. 12K Master Preview
    img_12k_path = PROJECT_ROOT / "assets" / "reference" / "results_12k" / "blackhole_12k_6912x12288_remacri.png"
    if img_12k_path.exists():
        print(f"Creating preview for 12K master: {img_12k_path.name}...")
        with Image.open(img_12k_path) as im:
            # Resize to 1080x1920 for web preview
            prev_12k = im.resize((1080, 1920), Image.Resampling.LANCZOS)
            prev_12k.save(showcase_dir / "preview_blackhole_12k.jpg", "JPEG", quality=88, optimize=True)
            print("  -> Saved preview_blackhole_12k.jpg")

    # 2. 4K Fidelity Master Preview
    fid_4k_path = PROJECT_ROOT / "output" / "experimental" / "v014" / "astronomy_test_4k_v014_09_deepzoom_fidelity.png"
    if fid_4k_path.exists():
        print("Creating preview for 4K Fidelity master...")
        with Image.open(fid_4k_path) as im:
            prev_fid = im.resize((1024, 1536), Image.Resampling.LANCZOS)
            prev_fid.save(showcase_dir / "preview_deepzoom_fidelity_4k.jpg", "JPEG", quality=88, optimize=True)
            print("  -> Saved preview_deepzoom_fidelity_4k.jpg")

    # 3. 4K Cinematic Master Preview
    cin_4k_path = PROJECT_ROOT / "output" / "experimental" / "v014" / "astronomy_test_4k_v014_10_deepzoom_cinematic.png"
    if cin_4k_path.exists():
        print("Creating preview for 4K Cinematic master...")
        with Image.open(cin_4k_path) as im:
            prev_cin = im.resize((1024, 1536), Image.Resampling.LANCZOS)
            prev_cin.save(showcase_dir / "preview_deepzoom_cinematic_4k.jpg", "JPEG", quality=88, optimize=True)
            print("  -> Saved preview_deepzoom_cinematic_4k.jpg")

    # 4. Copy Quad Panels into showcase
    crops_dir = PROJECT_ROOT / "output" / "experimental" / "v014" / "crops" / "matrix"
    if crops_dir.exists():
        for panel_p in crops_dir.glob("*_05_quad_panel.png"):
            target_p = showcase_dir / panel_p.name
            with Image.open(panel_p) as im:
                im.save(target_p, optimize=True)
        print(f"  -> Copied quad comparative panels to {showcase_dir}")

    # 5. Copy Difference Map
    diff_path = PROJECT_ROOT / "output" / "experimental" / "v014" / "diff_maps" / "master_v014_diff_fidelity_15x.png"
    if diff_path.exists():
        with Image.open(diff_path) as im:
            im.resize((1024, 1536), Image.Resampling.BOX).save(showcase_dir / "preview_diff_fidelity_15x.jpg", "JPEG", quality=85)
        print("  -> Saved preview_diff_fidelity_15x.jpg")

    print("Showcase previews generation complete!")

if __name__ == "__main__":
    generate_previews()
