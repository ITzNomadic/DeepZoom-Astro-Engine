"""
Deliverable Packaging Script for DeepZoom Astro Engine V0.1.2.
Packages:
1. DeepZoomStudio_AstroEngine_v0.1.2_AUDIT.zip
   - Complete codebase (src/, tests/, scripts/, docs/, benchmarks/v012/, requirements.txt, pyproject.toml, README.md)
   - Excludes virtualenvs, cache, binaries, and large masters.
2. DeepZoomStudio_AstroEngine_v0.1.2_VISUAL.zip
   - Full 4K masters (all 5 modes)
   - Difference maps (accumulated, amplified x15, per-layer)
   - 8 Strategic 1:1 crops, panels, and amplified difference maps
   - Manifest json
   - Module test crops
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def zip_audit_deliverable():
    zip_path = PROJECT_ROOT / "DeepZoomStudio_AstroEngine_v0.1.2_AUDIT.zip"
    print(f"Creating Audit Deliverable: {zip_path.name}")
    
    include_dirs = ["src", "tests", "docs", "scripts", "benchmarks/v012"]
    include_files = ["requirements.txt", "pyproject.toml", "README.md"]
    exclude_extensions = {".pyc", ".pyd", ".pyo"}
    exclude_names = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv"}

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        # Add root files
        for fname in include_files:
            fpath = PROJECT_ROOT / fname
            if fpath.exists():
                zipf.write(fpath, arcname=fname)
                file_count += 1
                print(f"  + {fname}")

        # Add directories
        for dname in include_dirs:
            dir_path = PROJECT_ROOT / dname
            if not dir_path.exists():
                continue
            for root, dirs, files in os.walk(dir_path):
                # Filter out exclude names
                dirs[:] = [d for d in dirs if d not in exclude_names]
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in exclude_extensions:
                        continue
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(PROJECT_ROOT)
                    zipf.write(full_p, arcname=str(rel_p))
                    file_count += 1

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Audit ZIP complete: {file_count} files, {size_mb:.2f} MB")
    return zip_path


def zip_visual_deliverable():
    zip_path = PROJECT_ROOT / "DeepZoomStudio_AstroEngine_v0.1.2_VISUAL.zip"
    print(f"\nCreating Visual Deliverable: {zip_path.name}")
    
    visual_root = PROJECT_ROOT / "output" / "experimental" / "v012"
    if not visual_root.exists():
        raise FileNotFoundError(f"Visual output directory not found: {visual_root}")

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # 1. 5 Full 4K Masters
        master_files = sorted(list(visual_root.glob("astronomy_test_4k_v012_*.png")))
        for mf in master_files:
            rel = mf.relative_to(PROJECT_ROOT)
            zipf.write(mf, arcname=str(rel))
            file_count += 1
            print(f"  + Master: {mf.name}")

        # 2. Difference Maps
        diff_dir = visual_root / "diff_maps"
        if diff_dir.exists():
            for df in sorted(list(diff_dir.glob("master_v012_*.png"))):
                rel = df.relative_to(PROJECT_ROOT)
                zipf.write(df, arcname=str(rel))
                file_count += 1
                print(f"  + Diff Map: {df.name}")

        # 3. Crops matrix & manifest
        matrix_dir = visual_root / "crops" / "matrix"
        if matrix_dir.exists():
            for cf in sorted(list(matrix_dir.iterdir())):
                if cf.is_file():
                    rel = cf.relative_to(PROJECT_ROOT)
                    zipf.write(cf, arcname=str(rel))
                    file_count += 1
            print(f"  + Crops Matrix ({len(list(matrix_dir.iterdir()))} items including manifest.json)")

        # 4. Other module verification crops
        for sub_crop in ["filaments", "stars"]:
            sc_dir = visual_root / "crops" / sub_crop
            if sc_dir.exists():
                for sf in sorted(list(sc_dir.glob("*.png"))):
                    rel = sf.relative_to(PROJECT_ROOT)
                    zipf.write(sf, arcname=str(rel))
                    file_count += 1
                print(f"  + Verification Crops: {sub_crop}/ ({len(list(sc_dir.glob('*.png')))} files)")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Visual ZIP complete: {file_count} files, {size_mb:.2f} MB")
    return zip_path


if __name__ == "__main__":
    print("================================================================================")
    print("DEEPZOOM ASTRO ENGINE V0.1.2 - DELIVERABLE PACKAGER")
    print("================================================================================")
    z_audit = zip_audit_deliverable()
    z_visual = zip_visual_deliverable()
    print("\nPackaging successful!")
    print(f"1. AUDIT ZIP:  {z_audit} ({z_audit.stat().st_size / 1024:.1f} KB)")
    print(f"2. VISUAL ZIP: {z_visual} ({z_visual.stat().st_size / (1024*1024):.2f} MB)")
