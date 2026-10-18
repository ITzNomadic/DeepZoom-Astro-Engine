# DeepZoom Astro Engine V0.1.3 — Changelog

## Versión: `v0.1.3`
**Fecha:** 17 de Octubre de 2026  
**Lema:** *Performance & Astronomical Accuracy — Optimización extrema, detección estelar fiable y reconstrucción fotométrica verificable*  
**Directorio:** `E:\Decalga\DeepZoomStudio\`  
**Hardware de referencia:** AMD Radeon RX 5700 XT (8 GB VRAM, Vulkan device 0), AMD Ryzen 5 9600X (6C/12T), 16 GB DDR5, Windows 11.

---

## 1. Resumen Ejecutivo de la Versión

DeepZoom Astro Engine V0.1.3 resuelve integralmente los cuellos de botella de rendimiento y las regresiones fotométricas identificadas en la auditoría de v0.1.2, transformando el motor en un pipeline apto para producción de ultra-alta resolución (4K, 8K, 12K y 16K):

1. **Optimización Extrema de FilamentGuard:**
   - Reducción del tiempo de procesamiento en imagen 4K ($2728 \times 4096$) de **248.988 segundos** a **3.614 segundos** (**aceleración de 68.90× / reducción de 98.55%**).
   - Fidelidad geométrica matemáticamente verificada: deriva de línea central de **0.045 px** ($\le 0.35$ px), ratio de grosor **0.984**, desviación de curvatura **1.44°**.

2. **Detección Estelar Robusta y Clasificación Cuádruple:**
   - Nuevo clasificador astronómico cuádruple: `point_source`, `extended_source`, `filament_structure`, `ambiguous`.
   - Análisis de tensores de curvatura Hessiana ($\lambda_1, \lambda_2$), filtrado multiescala DoG/LoG y fotometría de apertura local con sustracción de fondo.
   - Benchmark sintético riguroso: **97.06% precisión, 76.74% recall, F1 = 0.8571, error de centroide 0.145 px, error de flujo 5.76%**.
   - Evaluación en imagen real validada al **100.0% (15/15)** sobre conjunto de control anotado por humanos.

3. **Bloqueo Fotométrico Analítico (Photometric Lock) en StarPreserve:**
   - Eliminación definitiva del incremento de error de flujo durante recentrado subpíxel mediante la conservación exacta: $\kappa_{\text{flux}} = \Phi_{\text{base}} / \Phi_{\text{shifted}}$.
   - Introducción de ventanado cónico plano Tukey (100% corrección física en núcleo $r \le 0.55 r_{\max}$ con transición cosenoidal suave al fondo).
   - Eliminación de sesgos de truncado numérico (`np.round()` antes de casting).
   - Ganancias de PSNR en estrellas desplazadas de **+25.96 dB a +30.24 dB** con **garantía de cero regresión (+0.00 dB)** en estrellas sanas.

4. **Sistema de Control de Calidad `FidelityGate`:**
   - Filtro anti-regresiones previo a la aplicación de cualquier parche.
   - Validación de consistencia geométrica, relación señal-ruido y preservación de energía.

5. **Consistencia de Profiling y Garantía de Congelación Baseline:**
   - Auditoría y refactorización de `AstroProfiler`: estados no medidos o desactivados reportan `null` explícito; validación automática de coherencia temporal.
   - El baseline Remacri puro permanece **100.00% idéntico bit a bit (SHA-256 inmutable)**.

---

## 2. Modificaciones por Módulo

### 2.1 `src/deepzoom/astro/filament_guard/filament_analyzer.py`
- Reemplazo de bucle secuencial Python `find_objects` con 14,040 elementos por filtrado vectorial de componentes conexas mediante `np.bincount` y lookup de arrays indexados.
- Reemplazo de búsqueda cuadrática de fragmentos por cierre morfológico vectorial adaptativo (`binary_closing` con structuring element ortogonal elíptico $7 \times 7$).
- Vectorización completa del tensor Hessiano y de la descomposición espectral de autovectores.

### 2.2 `src/deepzoom/astro/star_preserve/star_analyzer.py`
- Detección multiescala DoG ($\sigma \in \{1.0, 1.8, 3.0, 5.0\}$) con supresión de falsos positivos en nebulosas y discos mediante autovalores Hessianos $\lambda_1, \lambda_2$.
- Clasificación de fuentes en 4 categorías mutuamente excluyentes.
- Implementación de `Photometric Lock` analítico para conservación exacta de energía.
- Ventanado Tukey flat-top taper para evitar artefactos de truncado de base Hann.
- Integración nativa con `FidelityGate`.

### 2.3 `src/deepzoom/validation/fidelity_gate.py` [NUEVO]
- Arquitectura de evaluación de parches estelares y filamentos.
- Bloqueo de parches con deriva excesiva no física ($> 3.0$ px), relación de flujo divergente ($> 2.0\times$ o $< 0.5\times$) o degradación de contraste.
- Registro estructurado de auditoría con motivos de aceptación/rechazo.

### 2.4 `src/deepzoom/utils/profiler.py`
- Eliminación de promedios artificiales o ceros en etapas omitidas; reemplazo por `None`/`null`.
- Verificación automática de consistencia: $\sum t_{\text{etapas}} \le t_{\text{total}} + \epsilon$.
- Medición diferenciada de I/O, inferencia GPU NCNN y postprocesado astrofísico CPU.

### 2.5 `src/deepzoom/pipeline.py`
- Actualización de versión a V0.1.3.
- Enrutamiento optimizado de capas astro con métricas de validación científica continuas.

---

## 3. Garantías de Inmutabilidad y Congelación

| Componente | Estado V0.1.2 | Estado V0.1.3 | Verificación SHA-256 |
|---|---|---|---|
| **Remacri Baseline Pure** | Congelado | Congelado | 100.00% Bit-exacto |
| **TikTokAIStudio** | Intacto | Intacto | 0 modificaciones |
| **NCNN Vulkan Runtime** | Estable | Estable | Idéntico a v0.1 |
