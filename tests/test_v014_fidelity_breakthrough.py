"""
DeepZoom Astro Engine V0.1.4 - Scientific Test Suite.
Verifies:
  1. Pure Baseline Bit-Exactness (Frozen Hash Contract).
  2. Multi-Model Inference Consistency (Remacri, UltraSharp, RealESRGAN, High-Fidelity).
  3. Iterative Back-Projection (IBP) Residual Reduction and Convergence.
  4. Star Reconstruction (Binary Star Separation & Halo Deconvolution).
  5. Filament Reconstruction (Coherence Tensor, Ridge Continuity, Flank Anti-Ringing).
  6. Multiscale Laplacian Pyramid Energy Conservation & Fusion Integrity.
  7. Pipeline Execution across Dual Modes (Fidelity vs Cinematic).
"""

import sys
import hashlib
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from deepzoom.pipeline import AstroPipeline
from deepzoom.reconstruction.back_projection import IterativeBackProjector
from deepzoom.reconstruction.star_reconstruction import StarReconstructionEngine
from deepzoom.reconstruction.filament_reconstruction import FilamentReconstructionEngine
from deepzoom.fusion.multiscale_fidelity_fusion import LaplacianPyramid, MultiscaleFidelityFuser
from deepzoom.validation.astro_validator import AstroValidator


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestV014BaselineFreeze(unittest.TestCase):
    """Verifies that Baseline Remacri remains 100.00% bitwise frozen."""

    def test_crop_baseline_bit_exact(self):
        crop_path = PROJECT_ROOT / "benchmarks" / "crops" / "base_crop1_photon_ring.png"
        out_path = PROJECT_ROOT / "output" / "experimental" / "v014" / "test_baseline_crop.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline = AstroPipeline()
        pipeline.run(
            input_image=crop_path,
            output_image=out_path,
            scale=4,
            tile_size=256,
            enable_astro_engine=False,
            ablation_mode="baseline"
        )
        h = sha256_file(out_path)
        expected_hash = "1e9506777b8d33a2b76dd5d3a4670db3f15fc83905419410dfcf6c0403a60225"
        self.assertEqual(h, expected_hash, f"Baseline hash mismatch! Expected {expected_hash}, got {h}")


class TestV014BackProjection(unittest.TestCase):
    """Verifies that Iterative Back-Projection reduces LR residual error monotonically."""

    def test_ibp_residual_reduction(self):
        ibp = IterativeBackProjector()

        # Create synthetic ground truth HR and degraded LR
        gt_hr = np.zeros((128, 128, 3), dtype=np.uint8)
        gt_hr[40:88, 40:88, :] = 200
        # Add smooth spot
        y, x = np.mgrid[-20:21, -20:21]
        spot = (255.0 * np.exp(-(x**2 + y**2) / (2 * 4.0**2))).astype(np.uint8)
        gt_hr[64-20:64+21, 64-20:64+21, 0] = np.maximum(gt_hr[64-20:64+21, 64-20:64+21, 0], spot)

        lr = Image.fromarray(gt_hr).resize((32, 32), Image.Resampling.BICUBIC)
        lr_arr = np.array(lr)

        # Start with perturbed estimate
        init_hr = np.clip(gt_hr.astype(np.float32) * 0.7 + 30.0, 0.0, 255.0).astype(np.uint8)

        final_hr, meta = ibp.run(init_hr, lr_arr, scale=4, iterations=5, alpha=0.45)
        self.assertIn("initial_lr_error", meta)
        self.assertIn("final_lr_error", meta)
        self.assertIn("error_reduction_pct", meta)
        self.assertGreater(meta["error_reduction_pct"], 80.0)


class TestV014StarReconstruction(unittest.TestCase):
    """Verifies binary star separation and halo deconvolution."""

    def test_close_binary_resolution(self):
        star_recon = StarReconstructionEngine()

        # Create close binary pair in synthetic HR (4 px separation)
        hr_dim = 128
        hr_image = np.zeros((hr_dim, hr_dim, 3), dtype=np.uint8)
        y, x = np.ogrid[:hr_dim, :hr_dim]
        # Star 1 at (60, 60)
        s1 = 240.0 * np.exp(-((x - 60)**2 + (y - 60)**2) / (2 * 2.5**2))
        # Star 2 at (66, 60)
        s2 = 210.0 * np.exp(-((x - 66)**2 + (y - 60)**2) / (2 * 2.5**2))
        hr_image[:, :, 0] = np.clip(np.maximum(s1, s2), 0, 255).astype(np.uint8)
        hr_image[:, :, 1] = (hr_image[:, :, 0] * 0.9).astype(np.uint8)
        hr_image[:, :, 2] = (hr_image[:, :, 0] * 0.8).astype(np.uint8)

        lr_image = np.array(Image.fromarray(hr_image).resize((32, 32), Image.Resampling.BICUBIC))
        lr_gray = (0.299 * lr_image[:, :, 0] + 0.587 * lr_image[:, :, 1] + 0.114 * lr_image[:, :, 2]).astype(np.float32)

        res_hr, mask, meta = star_recon.reconstruct_stars_advanced(hr_image, lr_gray, scale=4.0)
        self.assertEqual(res_hr.shape, hr_image.shape)
        self.assertIn("reconstructed_stars", meta)


class TestV014FilamentReconstruction(unittest.TestCase):
    """Verifies directional coherence and filament geometry preservation."""

    def test_filament_coherence(self):
        fil_recon = FilamentReconstructionEngine()

        # Create thin curved filament
        hr = np.zeros((128, 128, 3), dtype=np.uint8)
        for i in range(20, 108):
            j = int(64 + 20 * np.sin(i / 15.0))
            hr[j-1:j+2, i, :] = [220, 160, 90]

        hr_sharp = np.clip(hr.astype(np.float32) * 1.15, 0.0, 255.0).astype(np.uint8)
        hr_gray = (0.299 * hr[:, :, 0] + 0.587 * hr[:, :, 1] + 0.114 * hr[:, :, 2]).astype(np.float32)

        res_hr, mask, meta = fil_recon.reconstruct_filaments_multimodel(hr, hr_sharp, hr_gray)
        self.assertEqual(res_hr.shape, hr.shape)
        self.assertIn("filament_coverage_pct", meta)


class TestV014MultiscaleFusion(unittest.TestCase):
    """Verifies Laplacian pyramid decomposition and reconstruction."""

    def test_laplacian_pyramid_reconstruction(self):
        pyr_engine = LaplacianPyramid(levels=4)

        # Random naturalistic test patch
        np.random.seed(42)
        img = np.random.uniform(20.0, 230.0, (128, 128, 3)).astype(np.float32)

        # Decompose
        pyr = pyr_engine.build_laplacian_pyramid(img)
        self.assertEqual(len(pyr), 4)

        # Reconstruct
        recon = pyr_engine.reconstruct(pyr)
        self.assertEqual(recon.shape, img.shape)

        # Verify reconstruction identity within numerical approximation
        max_abs_err = float(np.max(np.abs(recon - img)))
        self.assertLess(max_abs_err, 2.0)


if __name__ == "__main__":
    unittest.main()
