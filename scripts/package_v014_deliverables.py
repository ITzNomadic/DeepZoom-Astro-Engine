"""
Packaging Script for DeepZoom Astro Engine V0.1.4 Deliverables:
1. DeepZoomStudio_AstroEngine_v0.1.4_AUDIT.zip:
   - Source code (src/)
   - Scripts (scripts/)
   - Tests (tests/)
   - Documentation & Reports (docs/, DeepZoomStudio_v0.1.4_RESEARCH_REPORT.md)
   - Benchmark reports (benchmarks/v014/)
   - Configs (config/)
   (Excludes binaries, models, venvs, and heavy master images)

2. DeepZoomStudio_AstroEngine_v0.1.4_VISUAL.zip:
   - 10 master images (astronomy_test_4k_v014_*.png)
   - Difference maps (diff_maps/)
   - 1:1 crops and quad panels (crops/matrix/)
   - Selection maps
   - Visual manifest.json
"""

import sys
import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def package_deliverables():
    print("=" * 80)
    print("PACKAGING DEEPZOOM ASTRO ENGINE V0.1.4 DELIVERABLES")
    print("=" * 80)

    dist_dir = PROJECT_ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    audit_zip_path = dist_dir / "DeepZoomStudio_AstroEngine_v0.1.4_AUDIT.zip"
    visual_zip_path = dist_dir / "DeepZoomStudio_AstroEngine_v0.1.4_VISUAL.zip"

    # -------------------------------------------------------------------------
    # 1. Package AUDIT ZIP
    # -------------------------------------------------------------------------
    print(f"\n[1/2] Creating AUDIT package: {audit_zip_path.name}...")
    audit_include_dirs = ["src", "scripts", "tests", "docs", "config"]
    audit_exclude_exts = {".bin", ".param", ".pyc", ".pyd", ".exe", ".dll", ".zip"}

    with zipfile.ZipFile(audit_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add dirs
        for dname in audit_include_dirs:
            dir_path = PROJECT_ROOT / dname
            if not dir_path.exists():
                continue
            for root, _, files in os.walk(dir_path):
                if "__pycache__" in root or ".pytest_cache" in root:
                    continue
                for f in files:
                    fp = Path(root) / f
                    if fp.suffix.lower() in audit_exclude_exts:
                        continue
                    rel_p = fp.relative_to(PROJECT_ROOT)
                    zipf.write(fp, arcname=str(rel_p))

        # Add benchmark JSONs
        bench_v014 = PROJECT_ROOT / "benchmarks" / "v014"
        if bench_v014.exists():
            for f in bench_v014.glob("*.json"):
                rel_p = f.relative_to(PROJECT_ROOT)
                zipf.write(f, arcname=str(rel_p))

        # Add root docs and research reports
        for f in PROJECT_ROOT.glob("*.md"):
            rel_p = f.relative_to(PROJECT_ROOT)
            zipf.write(f, arcname=str(rel_p))

    audit_mb = audit_zip_path.stat().st_size / (1024 * 1024)
    print(f"  -> Successfully created {audit_zip_path.name} ({audit_mb:.2f} MB)")

    # -------------------------------------------------------------------------
    # 2. Package VISUAL ZIP
    # -------------------------------------------------------------------------
    print(f"\n[2/2] Creating VISUAL package: {visual_zip_path.name}...")
    v014_out = PROJECT_ROOT / "output" / "experimental" / "v014"

    with zipfile.ZipFile(visual_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if v014_out.exists():
            # Add 4K master images
            for f in v014_out.glob("astronomy_test_4k_v014_*.png"):
                rel_p = f.relative_to(PROJECT_ROOT)
                zipf.write(f, arcname=str(rel_p))

            # Add diff maps
            diff_dir = v014_out / "diff_maps"
            if diff_dir.exists():
                for f in diff_dir.glob("*.png"):
                    rel_p = f.relative_to(PROJECT_ROOT)
                    zipf.write(f, arcname=str(rel_p))

            # Add crops and quad panels
            crops_dir = v014_out / "crops" / "matrix"
            if crops_dir.exists():
                for f in crops_dir.glob("*.*"):
                    rel_p = f.relative_to(PROJECT_ROOT)
                    zipf.write(f, arcname=str(rel_p))

            # Add selection maps
            for f in v014_out.glob("*selection_map*.png"):
                rel_p = f.relative_to(PROJECT_ROOT)
                zipf.write(f, arcname=str(rel_p))

    visual_mb = visual_zip_path.stat().st_size / (1024 * 1024)
    print(f"  -> Successfully created {visual_zip_path.name} ({visual_mb:.2f} MB)")
    print("\nPackaging complete!")

if __name__ == "__main__":
    package_deliverables()
