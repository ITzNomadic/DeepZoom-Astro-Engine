"""
High-Level Astronomical Super-Resolution Pipeline for DeepZoom Studio V0.1.4.
FIDELITY BREAKTHROUGH: Multi-Hypothesis Reconstruction, Iterative Back-Projection,
Advanced Star/Filament Modeling, and Multiscale Laplacian Pyramid Fusion.

Supports:
  1. Frozen Bit-Exact Baselines: 'baseline' (Remacri), 'ultrasharp', 'realesrgan', 'high_fidelity'.
  2. Inverse Consistency: 'remacri_plus_ibp'.
  3. Advanced Component Reconstructions: 'remacri_plus_star', 'remacri_plus_filament'.
  4. Multiscale Detail Fusion: 'multiscale_fusion'.
  5. Flagship Production Modes:
     - 'deepzoom_fidelity' (Strict observational physics, error-bounded, minimal hallucination)
     - 'deepzoom_cinematic' (Perceptual clarity, enhanced structural definition for deep zooms)
  6. Legacy V0.1.3 Ablation Modes: 'star_preserve_only', 'filament_guard_only', 'dark_sky_only', 'all_layers'.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union
import numpy as np
from PIL import Image

from deepzoom.core.config import DeepZoomConfig
from deepzoom.inference.ncnn_engine import NCNNEngine
from deepzoom.astro.star_preserve import StarPreserve
from deepzoom.astro.filament_guard import FilamentGuard
from deepzoom.astro.dark_sky import DarkSky
from deepzoom.astro.geometry_lock import GeometryLock
from deepzoom.fusion.adaptive_fuser import AdaptiveFuser
from deepzoom.validation.astro_validator import AstroValidator
from deepzoom.utils.profiler import AstroProfiler
from deepzoom.utils.image_io import load_image, get_image_info

# V0.1.4 Engines
from deepzoom.reconstruction.multi_model_engine import MultiModelEngine
from deepzoom.reconstruction.back_projection import IterativeBackProjector
from deepzoom.reconstruction.star_reconstruction import StarReconstructionEngine
from deepzoom.reconstruction.filament_reconstruction import FilamentReconstructionEngine
from deepzoom.fusion.multiscale_fidelity_fusion import MultiscaleFidelityFuser

PipelineMode = Literal[
    "baseline",
    "ultrasharp",
    "realesrgan",
    "high_fidelity",
    "remacri_plus_ibp",
    "remacri_plus_star",
    "remacri_plus_filament",
    "multiscale_fusion",
    "deepzoom_fidelity",
    "deepzoom_cinematic",
    "star_preserve_only",
    "filament_guard_only",
    "dark_sky_only",
    "all_layers"
]


class AstroPipeline:
    def __init__(self, config: Optional[DeepZoomConfig] = None):
        self.config = config or DeepZoomConfig()
        self.engine = NCNNEngine(self.config)
        self.star_preserve = StarPreserve()
        self.filament_guard = FilamentGuard()
        self.dark_sky = DarkSky()
        self.geometry_lock = GeometryLock()
        self.fuser = AdaptiveFuser(default_weight=0.5)
        self.validator = AstroValidator()

        # V0.1.4 Advanced Engines
        self.multi_model_engine = MultiModelEngine(self.config)
        self.back_projector = IterativeBackProjector()
        self.star_engine = StarReconstructionEngine()
        self.filament_engine = FilamentReconstructionEngine()
        self.multiscale_fuser = MultiscaleFidelityFuser(pyramid_levels=4)

    def run(
        self,
        input_image: Path,
        output_image: Path,
        model_name: str = "remacri-4x",
        target_res: Optional[str] = None,
        scale: int = 4,
        tile_size: Optional[int] = None,
        gpu_id: Optional[int] = None,
        enable_astro_engine: bool = False,
        ablation_mode: PipelineMode = "all_layers",
        astro_blend_weight: float = 0.5,
        ibp_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Executes super-resolution pipeline with V0.1.4 multi-hypothesis reconstruction.
        """
        profiler = AstroProfiler()
        profiler.start_total()

        in_path = Path(input_image).resolve()
        out_path = Path(output_image).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        effective_tile_size = tile_size or self.config.default_tile_size

        # ---------------------------------------------------------------------
        # CASE 1: Pure Frozen Baselines (Bit-Exact)
        # ---------------------------------------------------------------------
        is_pure_model = (not enable_astro_engine) or (ablation_mode in ["baseline", "ultrasharp", "realesrgan", "high_fidelity"])

        if is_pure_model:
            selected_model = model_name
            if ablation_mode == "ultrasharp":
                selected_model = "ultrasharp-4x"
            elif ablation_mode == "realesrgan":
                selected_model = "realesrgan-x4plus"
            elif ablation_mode == "high_fidelity":
                selected_model = "high-fidelity-4x"

            with profiler.measure("pure_ncnn_inference_seconds"):
                ncnn_res = self.engine.upscale(
                    input_image=in_path,
                    output_image=out_path,
                    model_name=selected_model,
                    target_res=target_res,
                    scale=scale,
                    tile_size=effective_tile_size,
                    gpu_id=gpu_id
                )

            profiler.end_total()
            out_info = get_image_info(out_path)
            return {
                **ncnn_res,
                "engine_version": "v0.1.4",
                "astro_engine_enabled": False,
                "mode": f"PURE_{selected_model.upper().replace('-', '_')}",
                "ablation_mode": ablation_mode,
                "profiling": profiler.get_summary(),
                "output_info": out_info
            }

        # ---------------------------------------------------------------------
        # CASE 2: Multi-Hypothesis and Advanced V0.1.4 Modes
        # ---------------------------------------------------------------------
        raw_baseline_out = out_path.with_suffix(".baseline.png")

        # Step 1: Base Remacri Inference
        with profiler.measure("pure_ncnn_inference_seconds"):
            self.engine.upscale(
                input_image=in_path,
                output_image=raw_baseline_out,
                model_name="remacri-4x",
                scale=scale,
                tile_size=effective_tile_size,
                gpu_id=gpu_id
            )

        with profiler.measure("io_read_seconds"):
            in_im = load_image(in_path)
            base_im = load_image(raw_baseline_out)

        with profiler.measure("preprocessing_seconds"):
            in_arr = np.array(in_im)
            in_gray = np.array(in_im.convert("L"))
            base_rgb = np.array(base_im)
            base_gray = np.array(base_im.convert("L"))
            eff_scale = float(base_im.size[0] / in_im.size[0])

        meta_artifacts = {}

        # Mode: Remacri + IBP
        if ablation_mode == "remacri_plus_ibp":
            with profiler.measure("back_projection_seconds"):
                fused_rgb, ibp_meta = self.back_projector.run(
                    hr_initial=base_rgb,
                    lr_original=in_arr,
                    scale=int(round(eff_scale)),
                    iterations=ibp_iterations,
                    alpha=0.45
                )
            meta_artifacts["ibp"] = ibp_meta

        # Mode: Remacri + Star Reconstruction
        elif ablation_mode == "remacri_plus_star":
            with profiler.measure("star_reconstruction_seconds"):
                fused_rgb, star_mask, star_meta = self.star_engine.reconstruct_stars_advanced(
                    baseline_rgb=base_rgb,
                    input_lr_gray=in_gray,
                    scale=eff_scale
                )
            meta_artifacts["star_reconstruction"] = star_meta

        # Mode: Remacri + Filament Reconstruction
        elif ablation_mode == "remacri_plus_filament":
            # Generate UltraSharp candidate
            ult_path = out_path.with_suffix(".ultrasharp.png")
            with profiler.measure("ultrasharp_inference_seconds"):
                self.engine.upscale(
                    input_image=in_path,
                    output_image=ult_path,
                    model_name="ultrasharp-4x",
                    scale=scale,
                    tile_size=effective_tile_size,
                    gpu_id=gpu_id
                )
            ult_rgb = np.array(load_image(ult_path))
            with profiler.measure("filament_reconstruction_seconds"):
                fused_rgb, fil_mask, fil_meta = self.filament_engine.reconstruct_filaments_multimodel(
                    candidate_remacri=base_rgb,
                    candidate_ultrasharp=ult_rgb,
                    image_gray=base_gray
                )
            meta_artifacts["filament_reconstruction"] = fil_meta

        # Modes: Flagship 'deepzoom_fidelity', 'deepzoom_cinematic', 'multiscale_fusion'
        elif ablation_mode in ["deepzoom_fidelity", "deepzoom_cinematic", "multiscale_fusion"]:
            fusion_mode = "cinematic" if ablation_mode == "deepzoom_cinematic" else "fidelity"

            # 1. Generate UltraSharp candidate
            ult_path = out_path.with_suffix(".ultrasharp.png")
            with profiler.measure("ultrasharp_inference_seconds"):
                self.engine.upscale(
                    input_image=in_path,
                    output_image=ult_path,
                    model_name="ultrasharp-4x",
                    scale=scale,
                    tile_size=effective_tile_size,
                    gpu_id=gpu_id
                )
            ult_rgb = np.array(load_image(ult_path))

            # 2. Generate IBP candidate
            with profiler.measure("back_projection_seconds"):
                ibp_rgb, ibp_meta = self.back_projector.run(
                    hr_initial=base_rgb,
                    lr_original=in_arr,
                    scale=int(round(eff_scale)),
                    iterations=ibp_iterations,
                    alpha=0.45
                )

            # 3. Generate Star Reconstruction candidate & mask
            with profiler.measure("star_reconstruction_seconds"):
                star_rgb, star_mask, star_meta = self.star_engine.reconstruct_stars_advanced(
                    baseline_rgb=base_rgb,
                    input_lr_gray=in_gray,
                    scale=eff_scale
                )

            # 4. Generate Filament mask & analysis
            with profiler.measure("filament_reconstruction_seconds"):
                geo_fil = self.filament_engine.analyze_geometry(base_gray)
                fil_mask = geo_fil["filament_mask"]

            # 5. Multiscale Laplacian Pyramid Fusion
            with profiler.measure("multiscale_fusion_seconds"):
                fused_rgb, fusion_res = self.multiscale_fuser.fuse(
                    candidate_remacri=base_rgb,
                    candidate_ultrasharp=ult_rgb,
                    candidate_ibp=ibp_rgb,
                    candidate_star_recon=star_rgb,
                    star_confidence_mask=star_mask,
                    filament_confidence_mask=fil_mask,
                    mode=fusion_mode
                )

            meta_artifacts["ibp"] = ibp_meta
            meta_artifacts["star_reconstruction"] = star_meta
            meta_artifacts["multiscale_fusion"] = fusion_res["meta"]

            # Save selection map
            sel_path = out_path.with_name(f"{out_path.stem}_selection_map.png")
            Image.fromarray(fusion_res["selection_map_rgb"]).save(sel_path)
            meta_artifacts["selection_map_path"] = str(sel_path)

        # Legacy V0.1.3 Fallback Modes
        else:
            in_stars = self.star_preserve.analyze_stars(in_gray)
            out_stars = self.star_preserve.analyze_stars(base_gray)
            star_recon_rgb, star_mask, _ = self.star_preserve.reconstruct_point_sources(
                baseline_rgb=base_rgb, input_stars=in_stars, output_stars=out_stars, scale_factor=eff_scale
            )
            fil_analysis = self.filament_guard.analyze_filaments(base_gray)
            fil_rep_rgb, fil_mask, _ = self.filament_guard.repair_filaments(baseline_rgb=base_rgb, analysis=fil_analysis)
            dark_analysis = self.dark_sky.analyze_background(base_gray)
            dark_rgb, dark_mask, _ = self.dark_sky.stabilize_vacuum(image_rgb=base_rgb, analysis=dark_analysis)

            fused_rgb, _ = self.fuser.fuse(
                baseline_rgb=base_rgb, mode=ablation_mode,
                star_reconstructed_rgb=star_recon_rgb, star_mask=star_mask,
                filament_repaired_rgb=fil_rep_rgb, filament_mask=fil_mask,
                dark_sky_rgb=dark_rgb, dark_sky_mask=dark_mask,
                weight=astro_blend_weight
            )

        # Step: Write final output image
        with profiler.measure("io_write_seconds"):
            Image.fromarray(fused_rgb).save(out_path, format="PNG")

        # Step: Scientific validation vs baseline
        with profiler.measure("validation_seconds"):
            val_metrics = self.validator.evaluate_all(base_rgb, fused_rgb)

        profiler.end_total()
        out_info = get_image_info(out_path)

        return {
            "status": "success",
            "engine_version": "v0.1.4",
            "mode": f"DEEPZOOM_{ablation_mode.upper()}",
            "ablation_mode": ablation_mode,
            "astro_engine_enabled": True,
            "output_info": out_info,
            "profiling": profiler.get_summary(),
            "validation_vs_baseline": val_metrics,
            "meta_artifacts": meta_artifacts
        }
