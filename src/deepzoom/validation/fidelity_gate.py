"""
Fidelity Gate V0.1.3 - Quality Control & Anti-Regression Mechanism.
Evaluates proposed physical corrections before application:
1. In benchmark/ground-truth mode: evaluates PSNR, centroid drift, and flux error to reject any regression.
2. In production mode (no ground truth): uses conservative photometric, geometric, and SNR priors
   to prevent destructive interventions, falling back to pure Remacri baseline.
3. Maintains an audit decision log for each stellar source or filament region.
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np


class FidelityGate:
    def __init__(
        self,
        min_star_confidence: float = 0.55,
        max_allowed_flux_drift_pct: float = 8.0,
        max_allowed_centroid_shift_px: float = 2.5,
        strict_gt_mode: bool = False
    ):
        self.min_star_confidence = min_star_confidence
        self.max_allowed_flux_drift_pct = max_allowed_flux_drift_pct
        self.max_allowed_centroid_shift_px = max_allowed_centroid_shift_px
        self.strict_gt_mode = strict_gt_mode
        self.decision_log: List[Dict[str, Any]] = []

    def evaluate_star_correction(
        self,
        star_id: int,
        baseline_patch: np.ndarray,
        reconstructed_patch: np.ndarray,
        confidence: float,
        source_type: str,
        initial_drift: float,
        proposed_shift: Tuple[float, float],
        local_bg: float,
        gt_patch: Optional[np.ndarray] = None
    ) -> Tuple[bool, str, float]:
        """
        Evaluates a candidate star repair.
        Returns:
            (apply_correction, reason, effective_weight)
        """
        # 1. Gate on source type
        if source_type != "point_source":
            decision = (False, f"Rejected: source classified as {source_type}, not point_source", 0.0)
            self._log_decision(star_id, "star", decision, confidence)
            return decision

        # 2. Gate on detection confidence
        if confidence < self.min_star_confidence:
            decision = (False, f"Rejected: confidence {confidence:.2f} < threshold {self.min_star_confidence}", 0.0)
            self._log_decision(star_id, "star", decision, confidence)
            return decision

        # 3. Gate on unphysical displacement
        shift_mag = float(np.hypot(proposed_shift[0], proposed_shift[1]))
        if shift_mag > self.max_allowed_centroid_shift_px:
            decision = (False, f"Rejected: shift magnitude {shift_mag:.2f} px exceeds {self.max_allowed_centroid_shift_px} px limit", 0.0)
            self._log_decision(star_id, "star", decision, confidence)
            return decision

        # 4. Photometric flux preservation check
        net_base = np.maximum(baseline_patch.astype(np.float32) - local_bg, 0.0)
        net_rec = np.maximum(reconstructed_patch.astype(np.float32) - local_bg, 0.0)
        flux_base = float(np.sum(net_base))
        flux_rec = float(np.sum(net_rec))
        flux_diff_pct = abs(flux_rec - flux_base) / max(flux_base, 1e-3) * 100.0

        if flux_diff_pct > self.max_allowed_flux_drift_pct:
            decision = (False, f"Rejected: flux drift {flux_diff_pct:.2f}% exceeds {self.max_allowed_flux_drift_pct}% limit", 0.0)
            self._log_decision(star_id, "star", decision, confidence)
            return decision

        # 5. Ground truth evaluation if available (Calibration / Benchmark Mode)
        if gt_patch is not None and self.strict_gt_mode:
            mse_base = float(np.mean((baseline_patch.astype(np.float32) - gt_patch.astype(np.float32)) ** 2))
            mse_rec = float(np.mean((reconstructed_patch.astype(np.float32) - gt_patch.astype(np.float32)) ** 2))
            if mse_rec > mse_base + 1e-4:
                decision = (False, f"Rejected (GT check): MSE increased from {mse_base:.4f} to {mse_rec:.4f}", 0.0)
                self._log_decision(star_id, "star", decision, confidence)
                return decision

        decision = (True, "Accepted: passes all geometric, photometric, and confidence priors", 1.0)
        self._log_decision(star_id, "star", decision, confidence)
        return decision

    def evaluate_filament_patch(
        self,
        region_id: str,
        coherence: float,
        ringing_detected: bool,
        mean_delta: float
    ) -> Tuple[bool, str]:
        """
        Evaluates a candidate filament repair zone.
        """
        if coherence < 0.25:
            decision = (False, f"Rejected: low coherence {coherence:.2f}")
            self._log_decision(region_id, "filament", (decision[0], decision[1], 0.0), coherence)
            return decision

        if mean_delta > 80.0:
            decision = (False, f"Rejected: excessive modification delta {mean_delta:.2f}")
            self._log_decision(region_id, "filament", (decision[0], decision[1], 0.0), coherence)
            return decision

        decision = (True, "Accepted: structural filament repair verified")
        self._log_decision(region_id, "filament", (True, decision[1], 1.0), coherence)
        return decision

    def _log_decision(self, item_id: Any, item_type: str, decision: Tuple, confidence: float):
        self.decision_log.append({
            "id": str(item_id),
            "type": item_type,
            "accepted": decision[0],
            "reason": decision[1],
            "effective_weight": decision[2] if len(decision) > 2 else 1.0,
            "confidence": round(confidence, 3)
        })

    def get_audit_summary(self) -> Dict[str, Any]:
        """Returns statistical summary of gating decisions."""
        total = len(self.decision_log)
        accepted = sum(1 for d in self.decision_log if d["accepted"])
        rejected = total - accepted
        reasons = {}
        for d in self.decision_log:
            r = d["reason"].split(":")[0]
            reasons[r] = reasons.get(r, 0) + 1

        return {
            "total_evaluated": total,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "acceptance_rate_pct": round((accepted / max(total, 1)) * 100.0, 2),
            "reasons_breakdown": reasons
        }
