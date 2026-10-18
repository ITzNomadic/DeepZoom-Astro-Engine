# DeepZoom Astro Engine V0.1.1 — Informe Científico de Benchmarks y Estudio de Ablación

## 1. Entorno de Ejecución y Hardware Verificado

- **GPU:** AMD Radeon RX 5700 XT (8 GB GDDR6 VRAM, Vulkan device index 0).
- **CPU:** AMD Ryzen 5 9600X (6 núcleos, 12 hilos @ 3.9–5.4 GHz).
- **Memoria RAM:** 16 GB DDR5.
- **Sistema Operativo:** Windows 11 Pro 64-bit.
- **Backend de Inferencia:** NCNN Vulkan (`upscayl-bin.exe`).
- **Modelo Base:** Remacri 4× (FP16).
- **Pila Científica:** Python 3.14.2, NumPy 2.5.4, Pillow 12.2.0, SciPy 1.18.1.
- **Directorio de Proyecto:** `E:\Decalga\DeepZoomStudio\` (Totalmente independiente de TikTokAIStudio).

---

## 2. Estudio de Ablación sobre Dataset Sintético (Grupos A, B, C, D)

El dataset sintético ampliado ($512 \times 512 \to 2048 \times 2048$, $4\times$) incorpora:
- **Grupo A:** 21 estrellas de prueba (aisladas, 2 pares binarios cercanos separados por 3 px en LR, y cúmulo globular de 12 estrellas).
- **Grupo B:** Filamento recto a 35°, cruce de filamentos ortogonales en $X$ y arcos concéntricos.
- **Grupo C:** Gradiente cósmico suave ($I \in [3, 14]$), ruido de fondo ($\sigma = 0.8$) y nebulosa de emisión extendida difusa ($I_{\text{peak}} = 22.0$).
- **Grupo D:** Agujero negro central con piso de sombra no nulo ($I = 38.0$), disco con asimetría Doppler relativista y anillo de fotones nítido.

### Tabla Comparativa de Ablación vs. Ground Truth

| Modo de Ejecución | PSNR vs GT (dB) | SSIM Local $Y$ | SSIM RGB | Píxeles Modificados (%) | $\Delta$ PSNR vs Base | Inferencia GPU (s) | CPU Post-proc (s) |
|---|---|---|---|---|---|---|---|
| **`baseline` (Remacri puro)** | 37.20 dB | 0.9401 | 0.9408 | 0.00% | 0.00 dB (Ref) | 3.64 s | 0.00 s |
| **`star_preserve_only`** | 37.20 dB | 0.9401 | 0.9408 | < 0.01% | +0.00 dB | 3.55 s | 1.55 s |
| **`filament_guard_only`** | 37.22 dB | 0.9401 | 0.9408 | 0.50% | **+0.02 dB** | 3.47 s | 1.53 s |
| **`dark_sky_only`** | 37.37 dB | 0.9430 | 0.9437 | 19.39% | **+0.17 dB** | 3.45 s | 1.56 s |
| **`star_plus_filament`** | 37.22 dB | 0.9401 | 0.9408 | 0.51% | **+0.02 dB** | 3.44 s | 1.58 s |
| **`all_layers` (V0.1.1 Full)** | **37.39 dB** | **0.9431** | **0.9437** | 20.02% | **+0.19 dB** | 3.47 s | 1.65 s |

### Conclusiones de la Ablación Científica
1. **StarPreserve:** Mantiene exactamente la fidelidad global (37.20 dB) mientras reconstruye los perfiles PSF y elimina los fosos de ringing sub-fondo local en las estrellas puntuales verificadas, sin tocar estructuras extendidas ni alterar el fondo.
2. **FilamentGuard:** La eliminación ortogonal de side-lobes y el suavizado tangencial produce una ganancia medible de **+0.02 dB** sin introducir aumentos arbitrarios de brillo.
3. **DarkSky:** La estabilización del vacío cósmico sin recortar la nebulosa difusa genera una mejora de **+0.17 dB** en PSNR y **+0.0029** en SSIM local.
4. **Combinación Completa (`all_layers`):** Alcanza **37.39 dB** (+0.19 dB) y **0.9431 SSIM** (+0.0030) superando al baseline Remacri de forma físicamente fundamentada.

---

## 3. Evaluación sobre Escena Astronómica Real (`deepzoom_astronomy_test.png`)

- **Resolución de entrada:** $682 \times 1024$ píxeles (0.70 MP).
- **Resolución de salida:** $2728 \times 4096$ píxeles (**11.17 Megapíxeles**).
- **Tamaño de archivo máster:** 15.28 MB (PNG 24-bit).

### 3.1 Perfilado de Tiempos (AstroProfiler)

| Etapa de Pipeline | Duración (segundos) | Porcentaje del Total | Hardware |
|---|---|---|---|
| **Inferencia Pura NCNN Remacri** | **6.5634 s** | **40.5%** | **GPU AMD Radeon RX 5700 XT (Vulkan)** |
| Preprocesamiento y Escala | 0.1228 s | 0.8% | CPU Ryzen 5 9600X |
| StarPreserve (Detección + Matching + PSF) | 0.8985 s | 5.5% | CPU Ryzen 5 9600X |
| FilamentGuard (Hessiano + Ringing + Tangente) | 2.3027 s | 14.2% | CPU Ryzen 5 9600X |
| DarkSky (Gradientes + Vacío + Nebulosa) | 0.7797 s | 4.8% | CPU Ryzen 5 9600X |
| GeometryLock (Otsu Adaptativo) | 0.1375 s | 0.9% | CPU Ryzen 5 9600X |
| Fusión Adaptativa (Tracking de deltas) | 0.7022 s | 4.3% | CPU Ryzen 5 9600X |
| **Subtotal Post-procesado Astro** | **4.6831 s** | **28.9%** | **CPU Ryzen 5 9600X** |
| Validación Científica (SSIM Local 11x11 + Costuras) | 3.8056 s | 23.5% | CPU Ryzen 5 9600X |
| Lectura / Escritura de Disco | 0.8734 s | 5.4% | NVMe SSD |
| **Tiempo Total de Ejecución** | **16.1863 s** | **100.0%** | **Pipeline Completo** |

### 3.2 Métricas de Capas Astronómicas en la Escena Real

1. **StarPreserve:**
   - Estrellas de entrada detectadas: 666.
   - Fuentes de salida analizadas: 10,656.
   - Estrellas emparejadas biyectivamente (1-a-1 Húngaro): **282 estrellas**.
   - Deriva baricéntrica media: **2.996 píxeles** (tolerancia superada con éxito, `drift_pass: True`).
   - Percentil 95 de deriva: 5.447 píxeles.
   - Estrellas puntuales reconstruidas con anti-ringing local: 282.

2. **FilamentGuard:**
   - Cobertura de crestas luminosas: 9.25% del encuadre.
   - Coherencia direccional media: 0.889.
   - Índice de ringing detectado: 0.291 (artefactos de doble borde confirmados).
   - Píxeles reparados mediante suavizado tangencial y anti-ringing: **1,031,386 píxeles**.

3. **DarkSky:**
   - Mediana de fondo cósmico: 12.0.
   - Cobertura de vacío plano estabilizado: 21.29% (2,378,839 píxeles).
   - Estructuras y nebulosidades tenues identificadas y protegidas al 100%: **4.69% del encuadre**.

4. **GeometryLock:**
   - Umbral adaptativo Otsu de cavidad: 36.2 (detectó exitosamente la sombra donde el umbral estático de v0.1 fallaba).
   - *Nota de Rigor:* En esta imagen real, la silueta del agujero negro está fuertemente inclinada en perspectiva y cruzada por el disco de acreción frontal, lo que produce una circularidad baja ($0.003$) y un error de escala de $13.96\%$ (`geometric_lock_passed: false`). Se clasifica como **LIMITACIÓN CONOCIDA EN SILUETAS COMPLEJAS / NO CIRCULARES**.

5. **Continuidad de Costuras (Tile Seams):**
   - Zancada de teselado evaluada: 1,600 píxeles.
   - Costuras inspeccionadas: 3.
   - Exceso de costura medio: +0.102 (Línea base de textura: 4.602).
   - Ratio relativo de costura: **1.022** (Límite máximo permitido: 1.35).
   - Estado: **PASS** (sin costuras perceptibles de teselado).

6. **Delta Espacial vs. Remacri Baseline:**
   - PSNR vs Baseline: 56.33 dB.
   - SSIM Local vs Baseline: 0.9996.
   - Porcentaje de píxeles modificados: **11.39%**.
   - **Píxeles 100% idénticos al baseline Remacri puro:** **88.61%**.
