# AdaptiveFusion V0.1.2 — Single-Weight Contract & Metric Synchronization

## 1. Architectural Motivation

In multi-stage astronomical enhancement, multiple specialized processors (StarPreserve, FilamentGuard, DarkSky) generate corrections for distinct physical phenomena.

In **DeepZoom v0.1.1**, two architectural bugs undermined fusion integrity:
1. **Double Weight Attenuation ($w^2$):** Both the individual sub-modules and the central fusion engine applied the blend weight $w$, causing an effective weighting of $w^2$ and nonlinear scaling when $w \ne 1.0$.
2. **Metric Desynchronization:** AdaptiveFuser and AstroValidator used slightly different rounding thresholds when classifying "modified pixels", causing discrepancies in benchmark reports.

**AdaptiveFusion v0.1.2** introduces a mathematically rigorous **Single-Weight Contract** and **Standardized Spatial Delta Synchronization**.

---

## 2. The Single-Weight Contract

### 2.1 Interface Specification
All astronomical modules (StarPreserve, FilamentGuard, DarkSky) adhere to a strict interface contract:
1. They return a **Pure Physical Repair** image $I_{\text{repaired}} \in [0, 255]$ representing the state of complete physical restoration ($w = 1.0$).
2. They return a **Continuous Spatial Confidence Mask** $M \in [0.0, 1.0]$ indicating where and with what astronomical confidence the repair applies.
3. Modules **MUST NOT** apply the user blend weight $w$.

### 2.2 Mathematical Fusion Formulation
The central `AdaptiveFuser` is the sole entity authorized to apply the global blend weight $w \in [0.0, 1.0]$:

$$I_{\text{fused}}(x, y) = I_{\text{base}}(x, y) + w \cdot \sum_{k \in \mathcal{K}} \alpha_k \cdot M_k(x, y) \odot \left(I_{\text{repaired}, k}(x, y) - I_{\text{base}}(x, y)\right)$$

where:
- $k \in \{\text{star}, \text{filament}, \text{dark\_sky}\}$
- $\alpha_k$ are priority partition weights ensuring $\sum_k \alpha_k \cdot M_k(x, y) \le 1.0$ at any pixel $(x, y)$.

### 2.3 Verification of Linearity
By definition, the delta image is:
$$\Delta(w) = I_{\text{fused}}(w) - I_{\text{base}} = w \cdot \Delta_{\text{pure}}$$
Testing at $w = 0.5$ versus $w = 1.0$:
$$\frac{\|\Delta(w=0.5)\|_1}{\|\Delta(w=1.0)\|_1} \equiv \mathbf{0.5000}$$
This linear scaling was empirically verified with $|0.5000 - 0.5000| < 10^{-4}$ in `tests/test_v012_fusion.py`.

---

## 3. Metric Synchronization with AstroValidator

To ensure that reporting across the entire pipeline is 100% consistent and audit-proof:
1. **Modified Pixel Condition:**
   $$\Delta(x, y) = \max_{c \in \{R, G, B\}} |I_{\text{fused}}^c(x, y) - I_{\text{base}}^c(x, y)| \ge 0.5$$
2. **Modified Pixel Count ($N_{\text{mod}}$):** Total pixels satisfying $\Delta(x, y) \ge 0.5$.
3. **Modification Percentage:**
   $$P_{\text{mod}} = \frac{N_{\text{mod}}}{H \times W} \times 100\%$$
4. **Mean Absolute Delta:**
   $$\mu_{\Delta} = \frac{1}{N_{\text{mod}}} \sum_{(x, y) \in \Omega_{\text{mod}}} \Delta(x, y)$$
5. **Distribution Percentiles:** Percentiles $p_{50}, p_{90}, p_{99}$ computed over the modified set $\Omega_{\text{mod}}$.

Both `AdaptiveFuser.get_fusion_stats()` and `AstroValidator.compute_spatial_delta()` use identical formulas, producing identical numbers down to the last decimal place.

---

## 4. Difference Map Generation Suite

To enable visual inspection of subtle subpixel corrections, `AdaptiveFuser` generates 4 distinct difference maps:
1. **Master Difference Map:** $|I_{\text{fused}} - I_{\text{base}}|$
2. **StarPreserve Layer Delta:** $M_{\text{star}} \odot |I_{\text{star\_recon}} - I_{\text{base}}|$
3. **FilamentGuard Layer Delta:** $M_{\text{filament}} \odot |I_{\text{filament\_repaired}} - I_{\text{base}}|$
4. **DarkSky Layer Delta:** $M_{\text{dark}} \odot |I_{\text{dark\_sky}} - I_{\text{base}}|$

Each map is saved as:
- **Raw Difference Map:** Physical pixel deltas $[0, 255]$
- **Amplified Difference Map ($\times 15$ Gain):** Clipped to $[0, 255]$ with warm colormap, revealing sub-intensity corrections, anti-ringing moats, and ridge bridges that are invisible to the naked eye.
