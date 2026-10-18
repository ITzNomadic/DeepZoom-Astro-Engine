"""
Image I/O, Hashing, and Comparison Utilities for DeepZoom Studio.
"""

import hashlib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

# Increase decompression limit to safely allow processing 150+ Megapixel images
Image.MAX_IMAGE_PIXELS = 300_000_000

def load_image(path: Path) -> Image.Image:
    """Loads an image and ensures RGB mode."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im

def get_image_info(path: Path) -> Dict[str, Any]:
    """Retrieves file size, dimensions, megapixels, and format."""
    with Image.open(path) as img:
        w, h = img.size
        fmt = img.format
        mode = img.mode
    sz_bytes = path.stat().st_size
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "width": w,
        "height": h,
        "dimensions": f"{w}x{h}",
        "megapixels": round((w * h) / 1_000_000, 2),
        "format": fmt,
        "mode": mode,
        "size_bytes": sz_bytes,
        "size_mb": round(sz_bytes / (1024 * 1024), 2)
    }

def calculate_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def compare_images_pixelwise(path1: Path, path2: Path) -> Dict[str, Any]:
    """
    Performs deterministic pixel-by-pixel comparison between two images.
    Returns mean absolute difference, max difference, and exact match boolean.
    """
    with Image.open(path1) as im1, Image.open(path2) as im2:
        if im1.size != im2.size:
            return {
                "identical_dimensions": False,
                "size1": im1.size,
                "size2": im2.size,
                "exact_match": False,
                "mean_abs_diff": None,
                "max_diff": None
            }
        
        arr1 = np.array(im1.convert("RGB"), dtype=np.int16)
        arr2 = np.array(im2.convert("RGB"), dtype=np.int16)
        
        diff = np.abs(arr1 - arr2)
        mean_diff = float(np.mean(diff))
        max_diff = int(np.max(diff))
        identical = bool(np.all(diff == 0))
        
        return {
            "identical_dimensions": True,
            "dimensions": im1.size,
            "exact_match": identical,
            "mean_abs_diff": mean_diff,
            "max_diff": max_diff,
            "matching_pixels_pct": float(np.mean(diff == 0) * 100.0)
        }
