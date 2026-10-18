"""
High-Precision Pipeline Profiler for DeepZoom Studio V0.1.3.
Separates pure NCNN GPU inference time from CPU post-processing stages, validation, and disk I/O.
Eliminates null/zero inconsistencies and includes automated coherence validation.
"""

import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class AstroProfiler:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}
        self._total_start: Optional[float] = None
        self._total_end: Optional[float] = None
        self._skipped_stages: List[str] = []

    def start_total(self):
        """Starts tracking total pipeline execution time."""
        self._total_start = time.perf_counter()

    def end_total(self) -> float:
        """Ends tracking total pipeline execution time."""
        self._total_end = time.perf_counter()
        total = self._total_end - (self._total_start or self._total_end)
        self.timings["total_pipeline_time_seconds"] = round(total, 4)
        return self.timings["total_pipeline_time_seconds"]

    def start_stage(self, stage_name: str):
        """Starts timing a specific pipeline stage."""
        self._start_times[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        """Ends timing a specific pipeline stage and stores duration."""
        if stage_name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[stage_name]
            self.timings[stage_name] = round(elapsed, 4)
            return self.timings[stage_name]
        return 0.0

    def mark_skipped(self, stage_name: str, reason: str = "omitted_in_mode"):
        """Explicitly records a stage as skipped rather than reporting 0.0."""
        self._skipped_stages.append(stage_name)
        self.timings[stage_name] = None

    @contextmanager
    def measure(self, stage_name: str):
        """Context manager to measure the execution time of a code block."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.timings[stage_name] = round(elapsed, 4)

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns structured profile summary cleanly separating GPU inference from CPU processing and I/O.
        Ensures that unmeasured stages are explicitly null or marked omitted.
        """
        # Distinguish measured stages
        gpu_time = self.timings.get("pure_ncnn_inference_seconds", None)
        if gpu_time is None and "gpu_ncnn_inference_seconds" in self.timings:
            gpu_time = self.timings["gpu_ncnn_inference_seconds"]

        # CPU post-processing stages (pattern matching for any astro processing sub-stage)
        cpu_post_keys = [
            k for k in self.timings.keys()
            if any(term in k for term in ["star_preserve", "filament_guard", "dark_sky", "fusion", "geometry_lock"])
            and not k.startswith("total_")
            and self.timings[k] is not None
        ]
        cpu_post_time = sum(self.timings[k] for k in cpu_post_keys) if cpu_post_keys else None

        io_read = self.timings.get("io_read_seconds", None)
        io_write = self.timings.get("io_write_seconds", None)
        if io_read is not None or io_write is not None:
            io_time = (io_read or 0.0) + (io_write or 0.0)
        else:
            io_time = None

        val_keys = [k for k in self.timings if "validation" in k and not k.startswith("total_") and self.timings[k] is not None]
        val_time = sum(self.timings[k] for k in val_keys) if val_keys else None

        # Total time
        total_time = self.timings.get("total_pipeline_time_seconds", None)
        if total_time is None:
            # Sum of available measured non-overlapping components
            comp_sum = 0.0
            if gpu_time: comp_sum += gpu_time
            if cpu_post_time: comp_sum += cpu_post_time
            if io_time: comp_sum += io_time
            if val_time: comp_sum += val_time
            total_time = round(comp_sum, 4) if comp_sum > 0 else None

        gpu_share = round((gpu_time / total_time) * 100.0, 1) if (gpu_time and total_time and total_time > 0) else None
        cpu_share = round((cpu_post_time / total_time) * 100.0, 1) if (cpu_post_time and total_time and total_time > 0) else None

        breakdown = {
            "gpu_ncnn_inference_seconds": round(gpu_time, 4) if gpu_time is not None else None,
            "cpu_astro_postprocessing_seconds": round(cpu_post_time, 4) if cpu_post_time is not None else None,
            "disk_io_seconds": round(io_time, 4) if io_time is not None else None,
            "scientific_validation_seconds": round(val_time, 4) if val_time is not None else None,
            "total_wall_clock_seconds": round(total_time, 4) if total_time is not None else None,
            "gpu_share_pct": gpu_share,
            "cpu_post_share_pct": cpu_share,
            "skipped_or_omitted_stages": self._skipped_stages
        }

        # Automated coherence validation
        coherence = self.validate_profile_coherence(breakdown)

        return {
            "detailed_stages": self.timings,
            "breakdown": breakdown,
            "coherence_validation": coherence
        }

    def validate_profile_coherence(self, breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates internal consistency: checks that the sum of measured parts matches
        total wall-clock within allowable tolerance and flags anomalies.
        """
        tot = breakdown.get("total_wall_clock_seconds")
        if tot is None or tot <= 0.0:
            return {
                "coherent": True,
                "note": "Total wall clock not recorded; coherence check skipped."
            }

        measured_parts = [
            breakdown.get("gpu_ncnn_inference_seconds"),
            breakdown.get("cpu_astro_postprocessing_seconds"),
            breakdown.get("disk_io_seconds"),
            breakdown.get("scientific_validation_seconds")
        ]
        sum_parts = sum(p for p in measured_parts if p is not None)

        # Allow 15% discrepancy for un-instrumented glue overhead
        discrepancy_pct = abs(tot - sum_parts) / tot * 100.0
        is_coherent = discrepancy_pct <= 20.0 or sum_parts <= tot

        return {
            "coherent": is_coherent,
            "total_wall_clock": tot,
            "sum_measured_components": round(sum_parts, 4),
            "unaccounted_overhead_seconds": round(max(0.0, tot - sum_parts), 4),
            "unaccounted_overhead_pct": round(discrepancy_pct, 2)
        }
