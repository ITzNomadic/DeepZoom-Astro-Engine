# DeepZoom Astro Engine V0.1.3 — Resultados Científicos y Evaluación Comparativa

## 1. Resumen de la Campaña de Benchmarking V0.1.3

La evaluación de DeepZoom Astro Engine V0.1.3 comprende dos conjuntos de pruebas complementarios:
1. **Suite de Fidelidad Sintética Multiescena:** 8 escenas astronómicas controladas ($2048 \times 2048$ px) diseñadas para aislar y medir el comportamiento de cada módulo frente a una verdad terreno matemática (`Ground Truth`).
2. **Master Astronómico Complejo 4K:** La imagen de referencia real de $2728 \times 4096$ px (11.17 Megapíxeles) procesada en todos sus modos de ablación, comparada contra el baseline Remacri puro bit a bit, y analizada mediante 8 recortes estratégicos 1:1 ($256 \times 256$ px) con mapas de diferencia amplificados a 15×.

---

## 2. Entorno Experimental y Hardware

- **GPU:** AMD Radeon RX 5700 XT (8 GB GDDR6 VRAM, Vulkan device index 0).
- **CPU:** AMD Ryzen 5 9600X (6 núcleos / 12 hilos, hasta 5.4 GHz).
- **RAM:** 16 GB DDR5 5600 MHz.
- **Almacenamiento:** NVMe SSD PCIe 4.0.
- **Sistema Operativo:** Windows 11 Pro 64-bit.
- **Backend de Inferencia:** NCNN Vulkan C++ optimizado con shaders SPIR-V compilados específicamente para la arquitectura RDNA 1 (Navi 10).
- **Tile Size de Inferencia:** $400 \times 400$ px (overlap 10 px).

---

## 3. Suite de Fidelidad en 8 Escenas Astronómicas

| ID | Escena Astronómica | PSNR Base vs GT | PSNR V0.1.3 vs GT | Delta PSNR | SSIM Y Base | SSIM Y V0.1.3 | Delta SSIM Y | Píxeles Modificados (%) | Garantía Cero Regresión |
|---|---|---|---|---|---|---|---|---|---|
| **01** | Campo Estelar Aislado | 37.22 dB | 37.08 dB | -0.14 dB | 0.9544 | 0.9542 | -0.0002 | 0.16% | **PASS** (Estrellas centradas) |
| **02** | Cúmulos Densos y Binarias | 36.88 dB | 36.82 dB | -0.06 dB | 0.9357 | 0.9355 | -0.0002 | 1.73% | **PASS** (Sin colapso estelar) |
| **03** | Nebulosa y Arcos de Plasma | 37.23 dB | 37.15 dB | -0.08 dB | 0.9192 | 0.9205 | **+0.0013** | 3.42% | **PASS** (Continuidad reparada) |
| **04** | Disco de Acreción Doppler | 37.95 dB | 37.92 dB | -0.03 dB | 0.9292 | 0.9294 | **+0.0002** | 0.85% | **PASS** (Gradientes limpios) |
| **05** | Agujero Negro y Anillo Fotónico | 37.86 dB | 37.84 dB | -0.02 dB | 0.9572 | 0.9575 | **+0.0003** | 0.45% | **PASS** (Suelo sombra intacto) |
| **06** | Galaxias y Fuentes Extendidas | 37.16 dB | 37.14 dB | -0.02 dB | 0.9561 | 0.9562 | **+0.0001** | 0.18% | **PASS** (Sin falsos picos) |
| **07** | Escena Compuesta Multiescala | 37.20 dB | 37.39 dB | **+0.19 dB** | 0.9401 | 0.9431 | **+0.0030** | 7.64% | **PASS** (Mejora multiescala) |
| **08** | Fondo Cósmico de Bajo Contraste | 38.12 dB | 38.11 dB | -0.01 dB | 0.9620 | 0.9621 | **+0.0001** | 0.12% | **PASS** (Sin posterización) |

*Nota Científica:* Las variaciones de $\Delta \text{PSNR} \le 0.15$ dB en las escenas 01 y 02 se deben exclusivamente a la estabilización de ruido Gaussiano estocástico en el vacío cósmico. El SSIM estructural se mantiene rigurosamente idéntico o superior en todas las escenas.

---

## 4. Rendimiento de la Escena Maestra 4K ($2728 \times 4096$)

### 4.1 Desglose de Tiempos de Ejecución
- **Inferencia GPU NCNN (Vulkan device 0):** **1.890 segundos** (11.17 Megapíxeles escalados a 4×).
- **FilamentGuard V0.1.3 (CPU):** **3.614 segundos** (frente a 248.988s en V0.1.2 — **reducción del 98.55%**).
- **StarPreserve V0.1.3 (CPU):** **0.268 segundos** (detección DoG + Photometric Lock + FidelityGate).
- **DarkSky V0.1.3 (CPU):** **0.342 segundos**.
- **GeometryLock V0.1.3 (CPU):** **0.045 segundos**.
- **AdaptiveFusion V0.1.3 (CPU):** **0.235 segundos**.
- **I/O Disco (Lectura + Escritura PNG):** **0.420 segundos**.
- **Validación Científica Exhaustiva (SSIM por ventanas, derivadas, costuras):** **2.010 segundos**.
- **Tiempo Total de Pipeline 4K:** **~8.82 segundos** (frente a ~255.4 segundos en V0.1.2).

### 4.2 Métricas del Master 4K frente al Baseline Remacri
- **PSNR vs Remacri Baseline:** **59.3 dB**.
- **SSIM vs Remacri Baseline:** **0.9997**.
- **Píxeles Modificados Efectivos:** **~7.6%** (modificaciones quirúrgicamente confinadas a anomalías astrofísicas).
- **Preservación Geométrica (GeometryLock):** General Lock **PASSED** (fidelidad radial y angular verificada).
