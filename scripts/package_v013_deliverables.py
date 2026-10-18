"""
Deliverable Packaging Script for DeepZoom Astro Engine V0.1.3.
Packages:
1. DeepZoomStudio_AstroEngine_v0.1.3_AUDIT.zip
   - Complete codebase (src/, tests/, docs/, scripts/, benchmarks/v013/, requirements.txt, pyproject.toml, README.md)
   - Excludes virtualenvs, cache, binaries, and heavy assets.
2. DeepZoomStudio_AstroEngine_v0.1.3_VISUAL.zip
   - Full 4K masters (all 5 modes)
   - Difference maps (accumulated, amplified x15, per-layer)
   - 8 Strategic 1:1 crops, panels, and amplified difference maps
   - Visual manifest.json
   - Fidelity scene outputs
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def zip_audit_deliverable():
    zip_path = PROJECT_ROOT / "DeepZoomStudio_AstroEngine_v0.1.3_AUDIT.zip"
    print(f"Creating Audit Deliverable: {zip_path.name}")

    include_dirs = ["src", "tests", "docs", "scripts", "benchmarks/v013"]
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
    zip_path = PROJECT_ROOT / "DeepZoomStudio_AstroEngine_v0.1.3_VISUAL.zip"
    print(f"\nCreating Visual Deliverable: {zip_path.name}")

    visual_root = PROJECT_ROOT / "output" / "experimental" / "v013"
    if not visual_root.exists():
        raise FileNotFoundError(f"Visual output directory not found: {visual_root}")

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # 1. 5 Full 4K Masters
        master_files = sorted(list(visual_root.glob("astronomy_test_4k_v013_*.png")))
        for mf in master_files:
            rel = mf.relative_to(PROJECT_ROOT)
            zipf.write(mf, arcname=str(rel))
            file_count += 1
            print(f"  + Master: {mf.name}")

        # 2. Difference Maps
        diff_dir = visual_root / "diff_maps"
        if diff_dir.exists():
            for df in sorted(list(diff_dir.glob("master_v013_*.png"))):
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

        # 4. Fidelity scene outputs
        fid_dir = visual_root / "fidelity_scenes"
        if fid_dir.exists():
            for ff in sorted(list(fid_dir.glob("*.png"))):
                rel = ff.relative_to(PROJECT_ROOT)
                zipf.write(ff, arcname=str(rel))
                file_count += 1
            print(f"  + Fidelity Scenes ({len(list(fid_dir.glob('*.png')))} images)")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Visual ZIP complete: {file_count} files, {size_mb:.2f} MB")
    return zip_path


if __name__ == "__main__":
    print("=" * 80)
    print("DEEPZOOM ASTRO ENGINE V0.1.3 - DELIVERABLE PACKAGER")
    print("=" * 80)
    z_audit = zip_audit_deliverable()
    z_visual = zip_visual_deliverable()
    print("=" * 80)
