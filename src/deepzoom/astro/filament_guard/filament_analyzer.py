"""
FilamentGuard V0.1.2 - Fine Luminous Structure, Ridge Continuity, and Anti-Ringing Reconstructor.
Implements:
1. Analytical Hessian eigenvalue and eigenvector decomposition (ridge normal v1 and tangent v2).
2. SIFT-style eigenvalue ratio filtering to cleanly distinguish 1D filaments from isotropic point sources.
3. Orthogonal anti-ringing suppression: detects and repairs artificial dark troughs flanking ridges along v1.
4. Tangential continuity bridging along v2 to heal micro-discontinuities and gaps.
5. Strict adherence to Single-Weight Contract: delivers pure physical corrections and normalized [0, 1] masks.
6. Quantitative geometric validation: centerline drift, thickness preservation (FWHM), curvature, and continuity.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import scipy.ndimage


class FilamentGuard:
    def __init__(
        self,
        ridge_threshold: float = 0.10,
        ringing_sensitivity: float = 1.8,
        coherence_threshold: float = 0.35,
        min_filament_length: int = 15
    ):
        self.ridge_threshold = ridge_threshold
        self.ringing_sensitivity = ringing_sensitivity
        self.coherence_threshold = coherence_threshold
        self.min_filament_length = min_filament_length

    def compute_hessian_eigen(
        self,
        image_gray: np.ndarray,
        sigma: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes 2nd order spatial derivatives and analytical Hessian eigenvalues/eigenvectors.
        Returns:
            lambda1: primary curvature eigenvalue (magnitude sorted, negative for bright ridge)
            lambda2: secondary curvature eigenvalue (along ridge tangent)
            v1: 2D unit vector orthogonal to ridge (direction of maximum curvature, y, x)
            v2: 2D unit vector tangent to ridge (direction along filament, y, x)
        """
        arr = image_gray.astype(np.float32)

        # 1st derivatives with Gaussian smoothing to avoid single-pixel noise spikes
        smoothed = scipy.ndimage.gaussian_filter(arr, sigma=sigma, mode='reflect')
        gy, gx = np.gradient(smoothed)

        # 2nd derivatives
        gyy, gyx = np.gradient(gy)
        gxy, gxx = np.gradient(gx)
        gxy_sym = 0.5 * (gyx + gxy)

        trace = gxx + gyy
        det = (gxx * gyy) - (gxy_sym ** 2)
        diff = np.sqrt(np.maximum((trace / 2.0) ** 2 - det, 0.0))

        e1 = (trace / 2.0) + diff
        e2 = (trace / 2.0) - diff

        swap_mask = np.abs(e2) > np.abs(e1)
        lambda1 = np.where(swap_mask, e2, e1)
        lambda2 = np.where(swap_mask, e1, e2)

        # Analytical unit eigenvector for lambda1 (v1: orthogonal normal across ridge)
        vy = gxx - lambda1
        vx = -gxy_sym
        norm = np.hypot(vy, vx)
        fallback_mask = norm < 1e-5

        vy2 = np.where(fallback_mask, -gxy_sym, vy)
        vx2 = np.where(fallback_mask, gyy - lambda1, vx)
        norm2 = np.maximum(np.hypot(vy2, vx2), 1e-6)

        v1_y = vy2 / norm2
        v1_x = vx2 / norm2

        # v2: unit tangent vector along the ridge (orthogonal to v1)
        v2_y = -v1_x
        v2_x = v1_y

        v1 = np.stack([v1_y, v1_x], axis=-1)  # (H, W, 2) in (y, x)
        v2 = np.stack([v2_y, v2_x], axis=-1)  # (H, W, 2) in (y, x)

        return lambda1, lambda2, v1, v2

    def analyze_filaments(
        self,
        image_gray: np.ndarray,
        sigma: float = 1.0
    ) -> Dict[str, Any]:
        """
        Analyzes ridge strength, orientation coherence, and detects double-edge ringing.
        Separates 1D filaments from isotropic point sources using coherence and eigenvalue ratios.
        """
        arr = image_gray.astype(np.float32)
        h, w = arr.shape
        lambda1, lambda2, v1, v2 = self.compute_hessian_eigen(arr, sigma=sigma)

        # Bright ridges have negative lambda1
        ridge_strength = np.maximum(-lambda1, 0.0)
        p99 = float(np.percentile(ridge_strength, 99.0)) if arr.size > 0 else 1.0
        norm_factor = max(p99, 1e-4)
        normalized_ridges = np.clip(ridge_strength / norm_factor, 0.0, 1.0)

        # Directional coherence: 1.0 for perfect 1D line/filament, 0.0 for isotropic circle
        coherence_num = (lambda1 - lambda2) ** 2
        coherence_den = (lambda1 + lambda2) ** 2 + 1e-4
        coherence = np.clip(coherence_num / coherence_den, 0.0, 1.0)

        # Candidate filament mask
        raw_mask = (normalized_ridges > self.ridge_threshold) & (coherence > self.coherence_threshold)

        # Filter out tiny isolated noise spots (< min_filament_length connected pixels)
        labeled, num_features = scipy.ndimage.label(raw_mask)
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())
            valid_mask = component_sizes >= self.min_filament_length
            valid_mask[0] = False  # background
            filament_mask = valid_mask[labeled]
        else:
            filament_mask = raw_mask

        filament_coverage_pct = float(np.mean(filament_mask) * 100.0)
        mean_coherence = float(np.mean(coherence[filament_mask])) if np.any(filament_mask) else 0.0

        # Ringing / double-edge detection
        gy, gx = np.gradient(arr)
        grad_mag = np.hypot(gx, gy)
        laplacian = lambda1 + lambda2
        p85_grad = float(np.percentile(grad_mag, 85.0)) if grad_mag.size > 0 else 1.0
        med_lap = float(np.median(np.abs(laplacian)))
        overshoot_mask = (laplacian > (self.ringing_sensitivity * max(med_lap, 1e-4)))
        high_grad_mask = (grad_mag > p85_grad)
        ringing_index = float(np.mean(overshoot_mask[high_grad_mask])) if np.any(high_grad_mask) else 0.0

        return {
            "filament_coverage_pct": round(filament_coverage_pct, 2),
            "mean_filament_coherence": round(mean_coherence, 3),
            "ringing_index": round(ringing_index, 4),
            "ringing_detected": bool(ringing_index > 0.22),
            "normalized_ridge_map": normalized_ridges,
            "coherence_map": coherence,
            "filament_mask": filament_mask,
            "v1": v1,
            "v2": v2,
            "lambda1": lambda1,
            "lambda2": lambda2
        }

    def repair_filaments(
        self,
        baseline_rgb: np.ndarray,
        analysis: Optional[Dict[str, Any]] = None,
        blend_weight: float = 1.0,
        pure_repair: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Applies curvature-guided physical repairs conforming to Single-Weight Contract:
        1. Tangential continuity bridging along v2 to heal breaks/notches.
        2. Orthogonal anti-ringing suppression along v1: lifts artificial dark troughs back to ambient.
        3. Delivers pure physical correction and normalized [0, 1] repair mask.

        Returns:
            repaired_rgb: uint8 array with pure physical repairs (or blended if pure_repair=False).
            repair_mask: float32 array in [0, 1] specifying spatial repair weights.
            metadata: dict with detailed repair diagnostics and metrics.
        """
        arr = baseline_rgb.astype(np.float32)
        h, w, c = arr.shape

        if analysis is None:
            gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            analysis = self.analyze_filaments(gray)

        filament_mask = analysis["filament_mask"]
        if not np.any(filament_mask):
            return baseline_rgb.copy(), np.zeros((h, w), dtype=np.float32), {
                "repaired_pixels": 0,
                "ringing_troughs_repaired": 0,
                "continuity_healed_pixels": 0
            }

        v1 = analysis["v1"]
        v2 = analysis["v2"]
        coherence = analysis["coherence_map"]
        normalized_ridges = analysis["normalized_ridge_map"]

        yy, xx = np.indices((h, w), dtype=np.int32)

        # Sample coordinates along tangent v2 (+/- 1.5 px)
        step_t = 1.5
        t_plus_y = np.clip(np.round(yy + step_t * v2[..., 0]).astype(np.int32), 0, h - 1)
        t_plus_x = np.clip(np.round(xx + step_t * v2[..., 1]).astype(np.int32), 0, w - 1)
        t_minus_y = np.clip(np.round(yy - step_t * v2[..., 0]).astype(np.int32), 0, h - 1)
        t_minus_x = np.clip(np.round(xx - step_t * v2[..., 1]).astype(np.int32), 0, w - 1)

        # Sample coordinates along normal v1:
        # Near trough (Gibbs depression): step_n_trough ~ 1.8 px
        step_n_trough = 1.8
        nt_plus_y = np.clip(np.round(yy + step_n_trough * v1[..., 0]).astype(np.int32), 0, h - 1)
        nt_plus_x = np.clip(np.round(xx + step_n_trough * v1[..., 1]).astype(np.int32), 0, w - 1)
        nt_minus_y = np.clip(np.round(yy - step_n_trough * v1[..., 0]).astype(np.int32), 0, h - 1)
        nt_minus_x = np.clip(np.round(xx - step_n_trough * v1[..., 1]).astype(np.int32), 0, w - 1)

        # Ambient floor outer sample: step_n_amb ~ 3.5 px
        step_n_amb = 3.5
        na_plus_y = np.clip(np.round(yy + step_n_amb * v1[..., 0]).astype(np.int32), 0, h - 1)
        na_plus_x = np.clip(np.round(xx + step_n_amb * v1[..., 1]).astype(np.int32), 0, w - 1)
        na_minus_y = np.clip(np.round(yy - step_n_amb * v1[..., 0]).astype(np.int32), 0, h - 1)
        na_minus_x = np.clip(np.round(xx - step_n_amb * v1[..., 1]).astype(np.int32), 0, w - 1)

        repaired = arr.copy()
        repair_mask = np.zeros((h, w), dtype=np.float32)

        # Mask of ringing moats flanking the filaments
        # Dilate filament mask by 2-3 pixels to reach flanking moats
        flank_zone = scipy.ndimage.binary_dilation(filament_mask, iterations=3) & (~filament_mask)

        ringing_repaired_count = 0
        continuity_healed_count = 0

        for ch in range(c):
            channel = arr[..., ch]

            # 1. Ridge continuity repair:
            # Directional tangent interpolation bridges micro-fractures along v2
            tangential_interp = 0.5 * (channel[t_plus_y, t_plus_x] + channel[t_minus_y, t_minus_x])
            # If current ridge pixel has dipped below its tangential neighbors, bridge it
            ridge_healed = np.maximum(channel, tangential_interp)

            # 2. Orthogonal anti-ringing repair on flanking moats:
            # Flanking background floor: use minimum of outer samples to prevent contamination by ridge
            ambient_floor = np.minimum(channel[na_plus_y, na_plus_x], channel[na_minus_y, na_minus_x])
            # If flanking pixel dips below ambient floor due to Gibbs ringing, clamp it
            trough_repaired = np.maximum(channel, ambient_floor)

            # Channel output combines ridge continuity on filament_mask and anti-ringing on flank_zone
            repaired_channel = channel.copy()
            repaired_channel = np.where(filament_mask, ridge_healed, repaired_channel)
            repaired_channel = np.where(flank_zone & (channel < ambient_floor), trough_repaired, repaired_channel)

            repaired[..., ch] = repaired_channel

        # 3. Inter-fragment continuity auto-bridging (connects fractured components within 12 px)
        closed_mask = scipy.ndimage.binary_closing(filament_mask, structure=np.ones((7, 7)))
        bridged_zone = closed_mask & (~filament_mask)
        if np.any(bridged_zone):
            for ch in range(c):
                smooth_ch = scipy.ndimage.gaussian_filter(repaired[..., ch], sigma=1.5)
                repaired[..., ch] = np.where(bridged_zone, np.maximum(repaired[..., ch], smooth_ch), repaired[..., ch])

        # Build repair mask [0, 1]
        # On filaments: proportional to coherence * normalized ridge strength
        repair_mask[filament_mask] = np.clip(coherence[filament_mask] * normalized_ridges[filament_mask], 0.0, 1.0)
        if np.any(bridged_zone):
            repair_mask[bridged_zone] = np.maximum(repair_mask[bridged_zone], 0.75)

        # On flanking moats: confidence based on depth of trough depression
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        ambient_gray = np.minimum(gray[na_plus_y, na_plus_x], gray[na_minus_y, na_minus_x])
        moat_depth = np.maximum(ambient_gray - gray, 0.0)
        moat_active = flank_zone & (moat_depth > 1.0)
        if np.any(moat_active):
            repair_mask[moat_active] = np.clip(moat_depth[moat_active] / 15.0, 0.1, 0.8)
            ringing_repaired_count = int(np.sum(moat_active))

        continuity_healed_count = int(np.sum(filament_mask & (repaired[..., 0] > arr[..., 0] + 1.0))) + int(np.sum(bridged_zone))

        # Clean Gaussian smoothing of the repair mask to avoid edge discontinuities
        repair_mask = scipy.ndimage.gaussian_filter(repair_mask, sigma=0.8, mode='reflect')
        repair_mask = np.clip(repair_mask, 0.0, 1.0).astype(np.float32)

        # Single-Weight contract:
        # If pure_repair is True, return the full pure physical repair without internal attenuation
        # If pure_repair is False (legacy standalone), apply blend_weight
        if not pure_repair:
            alpha = np.expand_dims(repair_mask * float(np.clip(blend_weight, 0.0, 1.0)), axis=-1)
            repaired = (1.0 - alpha) * arr + alpha * repaired

        final_rgb = np.clip(np.round(repaired), 0.0, 255.0).astype(np.uint8)
        repaired_pixels = int(np.sum(repair_mask > 0.02))

        return final_rgb, repair_mask, {
            "repaired_pixels": repaired_pixels,
            "ringing_troughs_repaired": ringing_repaired_count,
            "continuity_healed_pixels": continuity_healed_count,
            "mean_repair_weight": round(float(np.mean(repair_mask[repair_mask > 0.02])), 3) if repaired_pixels > 0 else 0.0
        }

    def evaluate_filament_geometry(
        self,
        ref_image_gray: np.ndarray,
        test_image_gray: np.ndarray,
        scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Quantitatively evaluates geometric properties of filaments:
        1. Centerline subpixel drift (mean and max).
        2. Thickness preservation (FWHM ratio).
        3. Curvature preservation (mean curvature deviation).
        4. Continuity (fragment count / connected components).
        5. Ringing depth reduction along v1.
        """
        ref = ref_image_gray.astype(np.float32)
        test = test_image_gray.astype(np.float32)

        # Analyze ref and test
        ref_analysis = self.analyze_filaments(ref)
        test_analysis = self.analyze_filaments(test)

        ref_mask = ref_analysis["filament_mask"]
        test_mask = test_analysis["filament_mask"]

        # 1. Continuity: connected components / fragment count
        _, ref_fragments = scipy.ndimage.label(ref_mask)
        _, test_fragments = scipy.ndimage.label(test_mask)

        # 2. Centerline drift
        # Find local maxima along v1 on test vs ref
        drift_px_list = []
        if np.any(ref_mask):
            v1_ref = ref_analysis["v1"]
            ref_pts = np.argwhere(ref_mask)
            # Sample subset for speed
            sample_step = max(1, len(ref_pts) // 200)
            sample_pts = ref_pts[::sample_step]

            for py, px in sample_pts:
                # 1D profile across normal v1 in ref and test with continuous spline sampling
                normal_vec = v1_ref[py, px]
                offsets = np.linspace(-3.0, 3.0, 25)
                coords = np.stack([py + offsets * normal_vec[0], px + offsets * normal_vec[1]])
                profile_ref = scipy.ndimage.map_coordinates(ref, coords, order=2)
                profile_test = scipy.ndimage.map_coordinates(test, coords, order=2)

                if np.ptp(profile_ref) > 4.0 and np.ptp(profile_test) > 4.0:
                    pk_ref = self._subpixel_peak(profile_ref, offsets)
                    pk_test = self._subpixel_peak(profile_test, offsets)
                    # Only calculate drift if the sampled point is centered on the ref ridge
                    if abs(pk_ref) <= 1.0:
                        drift = abs(pk_test - pk_ref)
                        drift_px_list.append(drift)

        mean_drift = float(np.mean(drift_px_list)) if drift_px_list else 0.0
        max_drift = float(np.max(drift_px_list)) if drift_px_list else 0.0

        # 3. Thickness (FWHM) preservation
        ref_fwhm_list = []
        test_fwhm_list = []
        if drift_px_list:
            for py, px in sample_pts:
                normal_vec = v1_ref[py, px]
                offsets_fwhm = np.linspace(-5.0, 5.0, 31)
                coords_fwhm = np.stack([py + offsets_fwhm * normal_vec[0], px + offsets_fwhm * normal_vec[1]])
                p_ref = scipy.ndimage.map_coordinates(ref, coords_fwhm, order=2)
                p_test = scipy.ndimage.map_coordinates(test, coords_fwhm, order=2)

                w_ref = self._estimate_fwhm(p_ref, offsets_fwhm)
                w_test = self._estimate_fwhm(p_test, offsets_fwhm)
                if w_ref > 0.5 and w_test > 0.5:
                    ref_fwhm_list.append(w_ref)
                    test_fwhm_list.append(w_test)

        mean_ref_fwhm = float(np.mean(ref_fwhm_list)) if ref_fwhm_list else 1.0
        mean_test_fwhm = float(np.mean(test_fwhm_list)) if test_fwhm_list else 1.0
        thickness_ratio = mean_test_fwhm / max(mean_ref_fwhm, 1e-3)

        # 4. Curvature deviation along tangent v2
        theta_ref = np.arctan2(ref_analysis["v2"][..., 0], ref_analysis["v2"][..., 1])
        theta_test = np.arctan2(test_analysis["v2"][..., 0], test_analysis["v2"][..., 1])
        common_filaments = ref_mask & test_mask
        if np.any(common_filaments):
            d_theta = np.abs(np.angle(np.exp(1j * (theta_test[common_filaments] - theta_ref[common_filaments]))))
            mean_curvature_dev_deg = float(np.mean(np.degrees(d_theta)))
        else:
            mean_curvature_dev_deg = 0.0

        # 5. Ringing moat depression depth reduction along normal v1
        flank_eval = scipy.ndimage.binary_dilation(ref_mask, iterations=5) & (~ref_mask)
        if np.any(flank_eval):
            # Outer ambient background floor
            outer_bg = ~scipy.ndimage.binary_dilation(ref_mask, iterations=7)
            bg_floor = float(np.median(ref[outer_bg])) if np.any(outer_bg) else float(np.min(ref))
            moat_ref = np.maximum(bg_floor - ref, 0.0)[flank_eval]
            moat_test = np.maximum(bg_floor - test, 0.0)[flank_eval]
            mean_moat_ref = float(np.mean(moat_ref))
            mean_moat_test = float(np.mean(moat_test))
            ringing_reduction_pct = float(max(0.0, (mean_moat_ref - mean_moat_test) / max(mean_moat_ref, 1e-4) * 100.0))
        else:
            mean_moat_ref = 0.0
            mean_moat_test = 0.0
            ringing_reduction_pct = 0.0

        return {
            "centerline_drift_mean_px": round(mean_drift, 3),
            "centerline_drift_max_px": round(max_drift, 3),
            "ref_mean_fwhm_px": round(mean_ref_fwhm, 2),
            "test_mean_fwhm_px": round(mean_test_fwhm, 2),
            "thickness_ratio": round(thickness_ratio, 3),
            "curvature_deviation_mean_deg": round(mean_curvature_dev_deg, 2),
            "ref_fragment_count": int(ref_fragments),
            "test_fragment_count": int(test_fragments),
            "ringing_moat_depth_ref": round(mean_moat_ref, 3),
            "ringing_moat_depth_test": round(mean_moat_test, 3),
            "ringing_reduction_pct": round(ringing_reduction_pct, 2)
        }

    def _subpixel_peak(self, profile: np.ndarray, offsets: np.ndarray) -> float:
        """Fits a local parabola around peak to find subpixel offset."""
        idx = int(np.argmax(profile))
        if idx == 0 or idx == len(profile) - 1:
            return float(offsets[idx])
        y0, y1, y2 = float(profile[idx - 1]), float(profile[idx]), float(profile[idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < 1e-4:
            return float(offsets[idx])
        delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
        step = float(offsets[1] - offsets[0])
        return float(offsets[idx] + delta * step)

    def _estimate_fwhm(self, profile: np.ndarray, offsets: np.ndarray) -> float:
        """Estimates full width at half maximum of a 1D peak."""
        p_min = float(np.min(profile))
        p_max = float(np.max(profile))
        amp = p_max - p_min
        if amp < 3.0:
            return 0.0

        half_max = p_min + 0.5 * amp
        above = np.where(profile >= half_max)[0]
        if len(above) >= 2:
            return float(offsets[above[-1]] - offsets[above[0]])
        return 0.0
