# DeepZoom Astro Engine V0.1.4 — Fase 0: Congelación y Estado Inicial

## 1. Declaración de Estado y Situación Inicial

Antes de emprender el salto arquitectónico de V0.1.4, se documenta formal y honestamente la situación de partida heredada de V0.1.3:

1. **Baseline Remacri:**
   - 100.00% idéntico bit a bit (SHA-256 inmutable):
     - Crop 1 (Photon Ring 2048x2048): `1e9506777b8d33a2b76dd5d3a4670db3f15fc83905419410dfcf6c0403a60225`
     - Master 8K (4608x8192): `5a3d71a38e43e40349360b42b6dc7720e8387808ea6a8c2a33ee51be347278bd`
   - Preservado en `benchmarks/baseline_reproduction_results.json`.

2. **Rendimiento V0.1.3:**
   - FilamentGuard fue acelerado 68.90× (de 248.988s a 3.614s en 4K).
   - Inferencia GPU NCNN: 1.89s en 4K (AMD RX 5700 XT 8GB VRAM).

3. **Limitaciones Críticas a Superar en V0.1.4:**
   - En la imagen artística principal (`deepzoom_astronomy_test.png`), `StarPreserve` no modificó ninguna estrella porque ninguna superó el umbral conservador de defectos tras el filtro de FidelityGate (`reconstructed_stars = 0`).
   - Las imágenes resultantes de las capas astro continuaban siendo perceptualmente casi idénticas a Remacri puro (diferencias de píxel $< 7.6\%$ y sólo en zonas de micro-corrección).
   - En benchmarks sintéticos con ruido aleatorio, las correcciones posteriores no lograban superar el límite intrínseco del modelo convolucional único.

4. **Copia de Seguridad:**
   - El código fuente completo de V0.1.3 ha sido respaldado en `backup/v013/src/`.

---

## 2. Nueva Hipótesis Científica de V0.1.4

Un único modelo (Remacri) no posee la verdad absoluta sobre todas las estructuras astronómicas (estrellas, filamentos, nebulosas, vacíos). En lugar de aplicar filtros cosméticos posteriores sobre la salida de un solo modelo, V0.1.4 introduce:
- **Reconstrucción Multi-Modelo:** Generación de múltiples hipótesis de alta resolución (`remacri-4x`, `ultrasharp-4x`, `realesrgan-x4plus`, `high-fidelity-4x`, `lanczos`).
- **Retroproyección Iterativa (IBP):** Restricción física de consistencia con la imagen de entrada original $D(HR) \approx LR$.
- **Reconstrucción Estelar Dedicada (PSF & Deconvolución):** Ajuste paramétrico de PSF (Gauss/Moffat) y desdoblamiento de estrellas dobles.
- **Reconstrucción Geométrica de Filamentos:** Tensores de curvatura multiescala con penalización de falsas uniones.
- **Fusión Multiescala en Pirámide Laplaciana:** Combinación adaptativa por frecuencias espaciales y regiones astrofísicas.
- **Dos Modos Claramente Diferenciados:** `fidelity` (rigor astrofísico conservador) y `cinematic` (impacto visual perceptual).
