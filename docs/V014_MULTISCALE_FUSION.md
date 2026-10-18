# DeepZoom Astro Engine V0.1.4 — Fase V: Fusión Multiescala en Pirámide Laplaciana

## 1. Arquitectura de Descomposición por Bandas de Frecuencia

La mezcla lineal directa en el espacio de color RGB ($I_{\text{fused}} = \alpha I_1 + (1-\alpha) I_2$) provoca emborronamiento y pérdida de contraste cuando los modelos presentan ligeras diferencias de fase o alineación subpíxel.

En V0.1.4 se implementa una **Pirámide Laplaciana de 4 niveles de frecuencia**:
1. **Banda $L_3$ (Estructura Global y Bajas Frecuencias):**
   - Anclada a la física consistente de IBP y Remacri.
   - Preserva los niveles absolutos de brillo, el fondo del vacío cósmico y los gradientes suaves sin derivas fotométricas.
2. **Banda $L_2$ (Detalle Medio):**
   - Estructuras intermedias de gas difuso, disco de acreción y brazos galácticos.
3. **Banda $L_1$ (Detalle Fino):**
   - Filamentos de plasma, arcos de choque e interfaces de ionización.
   - Gobernada por UltraSharp-4x y Retroproyección Iterativa (IBP).
4. **Banda $L_0$ (Micro-Detalle de Muy Alta Frecuencia):**
   - Núcleos estelares puntuales, perfiles de PSF y bordes afilados.
   - Inyección directa de candidatos especializados (`StarReconstructionEngine` y `FilamentReconstructionEngine`).

---

## 2. Modos de Reconstrucción: Fidelity vs. Cinematic

### Modo `fidelity`
- **Objetivo:** Máxima compatibilidad física y consistencia matemática con la observación de entrada.
- **Dominancia:** Gobernado principalmente por Retroproyección Iterativa (IBP) e interpolaciones conservadoras.
- **Resultado experimental:** Salto de **+10.79 dB de PSNR** sobre Remacri puro con preservación estricta de fuentes.

### Modo `cinematic`
- **Objetivo:** Máxima nitidez perceptual para movimientos de cámara y zoom profundo (10×).
- **Dominancia:** Equilibrio entre el fondo orgánico de Remacri-4x (~50%) y las crestas afiladas de UltraSharp-4x (~47.5%), con realce de micro-detalles en $L_0$.
- **Resultado experimental:** **+4.00 dB de PSNR** sobre Remacri puro con texturas vivas y filamentos altamente definidos.
