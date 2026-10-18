"""
DeepZoom Astro Engine V0.1.4 - Multi-Model Reconstruction Engine.
Generates and manages multi-hypothesis super-resolution candidates from identical input:
  1. Remacri-4x (Conservative baseline anchor)
  2. UltraSharp-4x (High-frequency edges & plasma filaments)
  3. RealESRGAN-x4plus (Smooth gradients & extended textures)
  4. High-Fidelity-4x (Astro-tuned balanced reconstruction)
  5. Lanczos-4x (Classical unbiased control baseline)

Guarantees:
  - Exact spatial alignment and matching output dimensions.
  - Consistent color space and dynamic range [0, 255].
  - Hardware acceleration on Vulkan GPU (AMD RX 5700 XT / GPU 0).
  - Reproducible SHA-256 cryptographic provenance tracking.
"""

import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image

from deepzoom.core.config import DeepZoomConfig
from deepzoom.inference.ncnn_engine import NCNNEngine
from deepzoom.utils.image_io import load_image, get_image_info


def sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class MultiModelEngine:
    """
    Orchestrates multi-hypothesis neural and analytical reconstructions.
    """
    AVAILABLE_MODELS = [
        "remacri-4x",
        "ultrasharp-4x",
        "realesrgan-x4plus",
        "high-fidelity-4x"
    ]

    def __init__(self, config: Optional[DeepZoomConfig] = None):
        self.config = config or DeepZoomConfig()
        self.engine = NCNNEngine(self.config)

    def generate_candidates(
        self,
        input_image: Path,
        output_dir: Path,
        models: Optional[List[str]] = None,
        scale: int = 4,
        tile_size: int = 400,
        gpu_id: Optional[int] = None,
        include_lanczos: bool = True
    ) -> Dict[str, Any]:
        """
        Generates candidate high-resolution reconstructions from a single input image.
        
        :param input_image: Path to LR input image.
        :param output_dir: Directory where candidate outputs will be saved.
        :param models: List of model names to run (default: Remacri, UltraSharp, RealESRGAN, High-Fidelity).
        :param scale: Scaling factor (default: 4).
        :param tile_size: NCNN tile size (default: 400).
        :param gpu_id: Vulkan GPU index (default: 0).
        :param include_lanczos: If True, also generates Lanczos4 interpolation as control.
        :return: Dictionary containing candidate file paths, arrays, timings, and metadata.
        """
        in_path = Path(input_image).resolve()
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            raise FileNotFoundError(f"Input image not found: {in_path}")

        selected_models = models or self.AVAILABLE_MODELS
        im_lr = load_image(in_path)
        w_lr, h_lr = im_lr.size
        target_w, target_h = w_lr * scale, h_lr * scale

        candidates = {}
        timings = {}

        # 1. Classical Lanczos Control Baseline
        if include_lanczos:
            t0 = time.time()
            lanczos_im = im_lr.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
            lanczos_path = out_dir / f"{in_path.stem}_candidate_lanczos.png"
            lanczos_im.save(lanczos_path, format="PNG")
            t_lanczos = time.time() - t0
            timings["lanczos-4x"] = round(t_lanczos, 3)
            candidates["lanczos-4x"] = {
                "name": "lanczos-4x",
                "type": "classical_interpolation",
                "file_path": str(lanczos_path),
                "sha256": sha256_file(lanczos_path),
                "dimensions": [target_w, target_h],
                "array": np.array(lanczos_im),
                "execution_seconds": round(t_lanczos, 3)
            }

        # 2. Neural NCNN Candidates
        for model_name in selected_models:
            out_candidate_path = out_dir / f"{in_path.stem}_candidate_{model_name}.png"
            t0 = time.time()
            res = self.engine.upscale(
                input_image=in_path,
                output_image=out_candidate_path,
                model_name=model_name,
                scale=scale,
                tile_size=tile_size,
                gpu_id=gpu_id
            )
            t_model = time.time() - t0
            timings[model_name] = round(t_model, 3)

            im_out = load_image(out_candidate_path)
            # Ensure RGB uint8
            if im_out.mode != "RGB":
                im_out = im_out.convert("RGB")
                im_out.save(out_candidate_path, format="PNG")

            out_arr = np.array(im_out)
            assert out_arr.shape[:2] == (target_h, target_w), (
                f"Dimension mismatch in {model_name}: expected ({target_h}, {target_w}), got {out_arr.shape[:2]}"
            )

            candidates[model_name] = {
                "name": model_name,
                "type": "neural_ncnn_vulkan",
                "file_path": str(out_candidate_path),
                "sha256": sha256_file(out_candidate_path),
                "dimensions": [target_w, target_h],
                "array": out_arr,
                "execution_seconds": round(t_model, 3)
            }

        return {
            "input_image": str(in_path),
            "input_dimensions": [w_lr, h_lr],
            "target_dimensions": [target_w, target_h],
            "scale_factor": scale,
            "candidates_count": len(candidates),
            "timings": timings,
            "candidates": candidates
        }

    def evaluate_candidate_divergence(
        self,
        candidates: Dict[str, Any],
        baseline_model: str = "remacri-4x"
    ) -> Dict[str, Any]:
        """
        Computes spatial divergence and metric differences between candidate models
        and the baseline anchor model.
        """
        if baseline_model not in candidates:
            raise KeyError(f"Baseline model {baseline_model} not in candidates")

        base_arr = candidates[baseline_model]["array"].astype(np.float32)
        divergences = {}

        for m_name, c_data in candidates.items():
            if m_name == baseline_model:
                continue
            arr = c_data["array"].astype(np.float32)
            abs_diff = np.abs(arr - base_arr)
            mean_diff = float(np.mean(abs_diff))
            max_diff = float(np.max(abs_diff))
            mod_pct = float(np.mean(np.any(abs_diff >= 1.0, axis=-1)) * 100.0)

            # High difference threshold (> 15 intensity levels)
            high_diff_pct = float(np.mean(np.any(abs_diff >= 15.0, axis=-1)) * 100.0)

            divergences[m_name] = {
                "mean_abs_difference": round(mean_diff, 2),
                "max_abs_difference": round(max_diff, 1),
                "modified_pixel_pct": round(mod_pct, 2),
                "significant_delta_pct": round(high_diff_pct, 2)
            }

        return divergences
