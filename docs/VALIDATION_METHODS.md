# AstroValidator V0.1.1 — Protocolo de Evaluación Métrica y Validación Científica

## 1. Introducción y Marco Teórico

La validación de modelos de superresolución astronómica requiere métricas físicamente significativas y reproducibles. DeepZoom Studio v0.1.1 sustituye las métricas empíricas globales por estándares rigurosos de visión computacional y fotometría estelar.

---

## 2. Índice de Similitud Estructural (SSIM Estándar Wang et al., 2004)

### 2.1 Formulación Matemática
Para dos imágenes $x$ e $y$, el índice SSIM local en la coordenada espacial $(i, j)$ se define como:
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

### 2.2 Parámetros de Implementación
- **Ventana de ponderación:** Núcleo Gaussiano simétrico circular de $11 \times 11$ píxeles con desviación estándar $\sigma = 1.5$, normalizado a $\sum w_i = 1$.
- **Rango dinámico ($L$):** $255.0$.
- **Constantes de estabilidad:**
  - $C_1 = (K_1 L)^2 = (0.01 \cdot 255)^2 = 6.5025$
  - $C_2 = (K_2 L)^2 = (0.03 \cdot 255)^2 = 58.5225$
- **Tratamiento de bordes:** Relleno por reflexión simétrica (`mode='reflect'`).
- **Evaluación:** Recorte del interior válido (excluyendo el radio de convolución de 5 píxeles) para evitar sesgos de contorno.
- **Espacio de color:**
  - **Luminancia ($Y$ según ITU-R BT.601):** $Y = 0.299R + 0.587G + 0.114B$ (métrica estándar reportada como `ssim_y`).
  - **Multicanal RGB:** Promedio de los mapas SSIM calculados independientemente en los canales R, G y B (`ssim_rgb`).

---

## 3. Emparejamiento Bipartito 1-a-1 de Catálogos Estelares

### 3.1 Algoritmo Húngaro (Linear Sum Assignment)
Dado el catálogo de entrada con $N_{\text{in}}$ fuentes proyectadas $(s \cdot x_{\text{in}}, s \cdot y_{\text{in}})$ y el catálogo de salida con $N_{\text{out}}$ fuentes $(x_{\text{out}}, y_{\text{out}})$:
1. Se calcula la matriz de costes de distancias euclidianas $D \in \mathbb{R}^{N_{\text{in}} \times N_{\text{out}}}$.
2. Se resuelve la asignación biyectiva óptima mediante el Algoritmo Húngaro (`scipy.optimize.linear_sum_assignment`).
3. Se aplica un umbral estricto de compuerta: $D_{ij} \le d_{\max} = 1.5 \cdot s$.
4. Las parejas que superan $d_{\max}$ se descartan como no coincidentes.

### 3.2 Métricas Fotométricas y de Posicionamiento
- **Precisión:** $P = \frac{N_{\text{matched}}}{N_{\text{out}}}$ (porcentaje de detecciones de salida que son estrellas reales).
- **Exhaustividad (*Recall*):** $R = \frac{N_{\text{matched}}}{N_{\text{in}}}$ (porcentaje de estrellas reales recuperadas).
- **Estadísticos de Deriva de Centroide:**
  - Media ($\mu_{\delta}$), Mediana ($M_{\delta}$), Máximo ($\max\delta$), percentil 75 ($P_{75}$), percentil 90 ($P_{90}$) y percentil 95 ($P_{95}$).
- **Conservación de Flujo Fotométrico:**
  $$\text{Ratio} = \frac{F_{\text{out}}}{F_{\text{in}} \cdot s^2}$$
- **Manejo Estricto de Catálogos sin Coincidencias:** Si $N_{\text{matched}} = 0$, todos los estadísticos de deriva y flujo son reportados explícitamente como `None` (nunca cero) y `drift_pass = False`.

---

## 4. Medición de Costuras de Teselas Referenciada por Control

Para imágenes procesadas en mosaico NCNN (*tiling*), las discontinuidades potenciales se evalúan en las coordenadas de teselado $x_k = k \cdot (\text{tile\_size} \cdot \text{scale})$:
1. **Salto en la costura:** $J_{\text{seam}} = \frac{1}{H} \sum_{y} |I(y, x_k) - I(y, x_k - 1)|$.
2. **Línea base de control local:** Se calcula el gradiente natural circundante a una distancia de control de $\pm 4$ píxeles:
   $$J_{\text{control}} = \frac{1}{2} [J(x_k - 4) + J(x_k + 4)]$$
3. **Exceso de costura:** $\Delta J = J_{\text{seam}} - J_{\text{control}}$.
4. **Ratio de costura:** $R_{\text{seam}} = \frac{J_{\text{control}} + \max(\Delta J, 0)}{J_{\text{control}}}$.
5. **Criterio de Aceptación:** $\max(\Delta J) < 2.5$ y $R_{\text{seam}} < 1.35$.

---

## 5. Perfilado de Rendimiento de Granularidad Fina

El componente `AstroProfiler` desagrega el tiempo de reloj en 10 etapas independientes:
1. `io_read_seconds`: Lectura y decodificación de la imagen de entrada y baseline.
2. `preprocessing_seconds`: Conversión a float32, escala de grises y extracción de escala.
3. `pure_ncnn_inference_seconds`: Tiempo de ejecución exclusivo del binario NCNN sobre la GPU AMD Radeon RX 5700 XT vía Vulkan.
4. `star_preserve_seconds`: Detección con filtro pasa-altos, matching Húngaro y reconstrucción local PSF.
5. `filament_guard_seconds`: Descomposición Hessiana, supresión de ringing y suavizado tangencial.
6. `dark_sky_seconds`: Análisis de gradiente, preservación de nebulosidad y estabilización de vacío.
7. `geometry_lock_seconds`: Segmentación adaptativa Otsu y verificación del horizonte.
8. `adaptive_fusion_seconds`: Fusión ponderada por capas y seguimiento de deltas espaciales.
9. `validation_seconds`: Cálculo de SSIM local, PSNR y análisis de costuras.
10. `io_write_seconds`: Codificación PNG y guardado en disco del máster final.
