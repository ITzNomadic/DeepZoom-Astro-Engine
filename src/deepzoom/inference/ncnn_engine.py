"""
NCNN Vulkan Inference Engine for DeepZoom Studio.
Directly communicates with Vulkan shaders to achieve hardware-accelerated super-resolution.
"""

import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from deepzoom.core.config import DeepZoomConfig
from deepzoom.core.exceptions import InferenceError, ModelNotFoundError
from deepzoom.utils.image_io import get_image_info

class NCNNEngine:
    def __init__(self, config: Optional[DeepZoomConfig] = None):
        self.config = config or DeepZoomConfig()
        self.runtime_path = self.config.ncnn_runtime_path
        self.models_dir = self.config.models_dir
        self.default_gpu_id = self.config.default_gpu_id
        self.default_tile_size = self.config.default_tile_size

    def upscale(
        self,
        input_image: Path,
        output_image: Path,
        model_name: Optional[str] = None,
        target_res: Optional[str] = None,
        scale: int = 4,
        tile_size: Optional[int] = None,
        resample_filter: str = "catmullrom",
        output_format: str = "png",
        gpu_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes NCNN Vulkan super-resolution inference on the target GPU.
        """
        input_image = Path(input_image).resolve()
        output_image = Path(output_image).resolve()

        if not input_image.exists():
            raise FileNotFoundError(f"Input image not found: {input_image}")

        output_image.parent.mkdir(parents=True, exist_ok=True)

        model = model_name or self.config.default_model
        # Verify model files exist in models directory
        self.config.get_model_path(model)

        gpu = gpu_id if gpu_id is not None else self.default_gpu_id
        tiles = tile_size if tile_size is not None else self.default_tile_size

        cmd = [
            str(self.runtime_path),
            "-i", str(input_image),
            "-o", str(output_image),
            "-m", str(self.models_dir),
            "-n", model,
            "-g", str(gpu),
            "-t", str(tiles),
            "-f", output_format
        ]

        if target_res:
            cmd.extend(["-r", f"{target_res}:{resample_filter}"])
        else:
            cmd.extend(["-s", str(scale)])

        t0 = time.time()
        # Capture raw bytes to prevent Windows CP1252 unicode decoding errors from emojis
        proc = subprocess.run(cmd, capture_output=True)
        t1 = time.time()
        elapsed = t1 - t0

        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace")
            raise InferenceError(
                f"NCNN Vulkan execution failed (code {proc.returncode}):\n{err_msg}\nCommand: {' '.join(cmd)}"
            )

        if not output_image.exists() or output_image.stat().st_size == 0:
            raise InferenceError(f"Output image was not generated or is empty: {output_image}")

        info = get_image_info(output_image)
        throughput = round(info["megapixels"] / max(elapsed, 0.001), 2)

        return {
            "status": "success",
            "model_used": model,
            "gpu_id": gpu,
            "tile_size": tiles,
            "target_resolution": target_res or f"{scale}x",
            "execution_time_seconds": round(elapsed, 2),
            "throughput_mp_per_sec": throughput,
            "output_info": info
        }
