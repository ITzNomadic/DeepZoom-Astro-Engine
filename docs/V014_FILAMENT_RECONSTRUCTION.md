# DeepZoom Astro Engine V0.1.4 — Fase IV: Motor de Reconstrucción de Filamentos

## 1. Problema de la Clausura Morfológica Ingenua

En versiones anteriores, la reparación de filamentos dependía de cierres morfológicos binarios (`binary_closing`). Aunque esta operación cerraba pequeñas micro-brechas, presentaba un riesgo astronómico:
- Si dos filamentos paralelos de plasma o brazos espirales pasaban a 5-6 píxeles de distancia, el cierre morfológico los unía artificialmente en una única banda ancha ("falsa fusión").
- Si un filamento terminaba abruptamente en el vacío, el algoritmo intentaba extenderlo hacia el ruido vecino.

En V0.1.4 se sustituye el cierre morfológico por un **análisis tensorial de estructura y curvatura direccional**:
- **Tensor de Estructura Multiescala:** Mide la coherencia direccional $C \in [0, 1]$ y extrae la orientación tangencial $v_2 = (-\sin \theta, \cos \theta)$.
- **Tensor Hessiano:** Mide la curvatura ortogonal $\lambda_1$ sobre la cresta del filamento.
- **Trazado de Líneas Centrales Subpíxel:** Identifica la posición de máxima intensidad a lo largo de la normal sin engrosar la estructura.

---

## 2. Descubrimiento en Comparación de Modelos

Al evaluar candidatos en la escena sintética con filamentos (`03_nebula_filaments_gt_2048.png`):
- **Remacri-4x:** PSNR = 37.23 dB | SSIM = 0.9192
- **UltraSharp-4x:** **PSNR = 37.83 dB (+0.60 dB) | SSIM = 0.9670 (+0.0478)**

UltraSharp-4x supera de forma neta a Remacri en filamentos y arcos delgados de plasma debido a su función de pérdida orientada a altas frecuencias espaciales, que preserva los bordes afilados sin el desenfoque suave característico de Remacri.

---

## 3. Integración en la Fusión de V0.1.4

Para evitar que la mezcla espacial directa difumine crestas con sutiles diferencias de fase subpíxel:
1. En la cresta del filamento, la componente de detalle de alta frecuencia se extrae de **UltraSharp-4x o IBP**.
2. En los flancos del filamento, se aplica el **clamping anti-ringing** para asegurar que el pozo de Gibbs no descienda por debajo del fondo local astronómico.
3. El fondo difuso y las nubes de gas continuas se preservan desde el ancla de baja frecuencia de Remacri.
