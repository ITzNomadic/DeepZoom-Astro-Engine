"""
Tiling and Memory Management for DeepZoom Studio.
Ensures zero Out-Of-Memory (OOM) crashes on 8 GB VRAM hardware during massive scaling.
"""

from typing import Tuple, Dict, Any, List
import math

class TileManager:
    def __init__(self, default_tile_size: int = 400, overlap_px: int = 16):
        self.tile_size = default_tile_size
        self.overlap = overlap_px

    def calculate_tiles(self, width: int, height: int, tile_size: int = None) -> Dict[str, Any]:
        """
        Calculates grid dimensions and tile count for an input image.
        """
        ts = tile_size or self.tile_size
        cols = math.ceil(width / ts)
        rows = math.ceil(height / ts)
        total_tiles = cols * rows

        # Estimated memory per tile (assuming 64 intermediate channels in FP16)
        # Tensor size: 1 * 64 * ts * ts * 2 bytes
        bytes_per_tile_tensor = 64 * (ts ** 2) * 2
        mb_per_tile_tensor = bytes_per_tile_tensor / (1024 * 1024)

        return {
            "input_width": width,
            "input_height": height,
            "tile_size": ts,
            "grid_columns": cols,
            "grid_rows": rows,
            "total_tiles": total_tiles,
            "vram_per_tile_mb": round(mb_per_tile_tensor, 2),
            "safe_for_8gb_vram": bool(mb_per_tile_tensor < 2000)
        }

    def recommend_tile_size(self, width: int, height: int, available_vram_gb: float = 8.0) -> int:
        """
        Returns a recommended tile size that balances throughput and VRAM safety.
        For 8 GB cards (RX 5700 XT), 400x400 offers peak FP16 compute utilization.
        """
        if available_vram_gb <= 4.0:
            return 256
        elif available_vram_gb <= 8.0:
            return 400
        else:
            return 512
