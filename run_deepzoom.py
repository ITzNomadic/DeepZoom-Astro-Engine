"""
DeepZoom Astro Engine - Command Line Interface (CLI).
Enables single-command super-resolution for astronomical imagery using Vulkan NCNN.

Examples:
    # High-Fidelity Scientific Reconstruction (Observational IBP consistency + PSF):
    python run_deepzoom.py -i samples/astronomical/deepzoom_astronomy_test.png -o output/result_fidelity.png --mode fidelity

    # Cinematic Enhanced Reconstruction (Enhanced filament high-frequency edge energy):
    python run_deepzoom.py -i samples/astronomical/deepzoom_astronomy_test.png -o output/result_cinematic.png --mode cinematic

    # Pure Remacri Baseline:
    python run_deepzoom.py -i samples/astronomical/deepzoom_astronomy_test.png -o output/result_remacri.png --mode baseline
"""

import sys
import argparse
from pathlib import Path

# Add src to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from deepzoom.pipeline import AstroPipeline
from deepzoom.core.config import DeepZoomConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepZoom Astro Engine V0.1.4 - Scientific Multi-Model Super-Resolution (Vulkan/NCNN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", type=str, help="Path to input low-resolution image.")
    parser.add_argument("-o", "--output", type=str, help="Path to save the super-resolved output image.")
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="fidelity",
        choices=[
            "fidelity",
            "cinematic",
            "multiscale_fusion",
            "remacri_plus_ibp",
            "remacri_plus_star",
            "remacri_plus_filament",
            "baseline",
            "ultrasharp",
            "realesrgan",
            "high_fidelity"
        ],
        help="Reconstruction pipeline mode."
    )
    parser.add_argument("-s", "--scale", type=int, default=4, help="Upscale factor (e.g. 4 for 4x).")
    parser.add_argument("-t", "--tile-size", type=int, default=400, help="Tile size for VRAM-constrained GPU inference.")
    parser.add_argument("-g", "--gpu-id", type=int, default=0, help="Vulkan GPU device ID (0 for primary discrete GPU).")
    parser.add_argument("--ibp-iter", type=int, default=5, help="Number of Iterative Back-Projection iterations.")
    parser.add_argument("--detect-gpu", action="store_true", help="Print available Vulkan GPUs and exit.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.detect_gpu:
        print("Detecting available Vulkan GPUs...")
        config = DeepZoomConfig()
        devices = config.detect_vulkan_devices()
        for dev_id, dev_name in devices.items():
            print(f"  [Device {dev_id}] {dev_name}")
        return 0

    if not args.input or not args.output:
        print("Error: Both --input (-i) and --output (-o) are required.")
        print("Run 'python run_deepzoom.py --help' for usage details.")
        return 1

    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()

    if not in_path.exists():
        print(f"Error: Input image does not exist: {in_path}")
        return 1

    print("=" * 70)
    print("DEEPZOOM ASTRO ENGINE - SUPER-RESOLUTION INFERENCE")
    print("=" * 70)
    print(f"Input:       {in_path}")
    print(f"Output:      {out_path}")
    print(f"Mode:        {args.mode.upper()}")
    print(f"Scale:       {args.scale}x")
    print(f"Tile Size:   {args.tile_size} px")
    print(f"GPU ID:      {args.gpu_id}")
    print("-" * 70)

    pipeline = AstroPipeline()

    enable_engine = args.mode not in ["baseline", "ultrasharp", "realesrgan", "high_fidelity"]
    pipeline_mode = args.mode
    if args.mode == "fidelity":
        pipeline_mode = "deepzoom_fidelity"
    elif args.mode == "cinematic":
        pipeline_mode = "deepzoom_cinematic"

    try:
        res = pipeline.run(
            input_image=in_path,
            output_image=out_path,
            scale=args.scale,
            tile_size=args.tile_size,
            gpu_id=args.gpu_id,
            enable_astro_engine=enable_engine,
            ablation_mode=pipeline_mode,
            ibp_iterations=args.ibp_iter
        )
        print("\n" + "=" * 70)
        print("RECONSTRUCTION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Saved:       {out_path}")
        info = res.get("output_info", {})
        print(f"Dimensions:  {info.get('width', 'N/A')}x{info.get('height', 'N/A')} ({info.get('megapixels', 'N/A')} MP)")
        prof = res.get("profiling", {})
        print(f"Total Time:  {prof.get('total_pipeline_seconds', 'N/A')}s")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
