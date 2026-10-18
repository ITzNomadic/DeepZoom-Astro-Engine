# DeepZoom Astro Engine V0.1.3 — Limitaciones Técnicas y Fronteras Científicas

En cumplimiento de los principios de transparencia y rigor científico, este documento detalla las limitaciones conocidas de DeepZoom Astro Engine V0.1.3, delimitando con precisión qué problemas están resueltos y cuáles constituyen fronteras de investigación activa para versiones futuras.

---

## 1. Detección y Separación en Cúmulos Ultra-Densos y Binarias Muy Cerradas

### Estado Actual:
En V0.1.3, el clasificador morfológico cuádruple evita activamente la deformación de sistemas binarios o cúmulos al categorizarlos como `extended_source` o `ambiguous` cuando la elipticidad supera $0.38$ o el ratio de autovalores Hessianos no es isotrópico.

### Limitación Científica:
- **Límite de Rayleigh Monocuadro:** Cuando dos fuentes estelares presentan una separación angular en la imagen de entrada menor a $1.2 \cdot \text{FWHM}_{\text{input}}$ ($\approx 2.5$ a $3.0$ píxeles), las alas de difracción se solapan coherentemente, formando un único lóbulo elongado.
- El algoritmo no intenta desdoblar de forma heurística dos centros de gravedad cuando no existe una bifurcación de intensidad medible, priorizando la no-invención de estrellas sobre la resolución forzada.
- **Recall en Cúmulos Densos:** En cúmulos con densidades $> 100$ estrellas por $100 \times 100$ px, el recall de estrellas individuales aisladas desciende al **$58.3\%$**, debido a la exclusión defensiva impuesta por `FidelityGate`.

---

## 2. Estimación de Fondo Anular en Frentes de Choque Escarpados

### Estado Actual:
La fotometría de apertura utiliza un anillo externo concéntrico ($r_{\text{in}} = 2.0 \cdot \sigma$, $r_{\text{out}} \in [2.5 \cdot \sigma, 4.0 \cdot \sigma]$) y la mediana robusta para aislar el fondo.

### Limitación Científica:
- En zonas de choque astrofísico abrupto (frentes de ionización HII o bordes afilados de nubes de polvo oscuras con saltos de intensidad $\Delta I > 50$ en menos de 5 píxeles), el anillo de fondo puede abarcar simultáneamente el vacío oscuro y la nube brillante.
- Aunque la mediana (`np.median`) y la desviación absoluta mediana (`MAD`) mitigan la asimetría, un gradiente fuertemente anisotrópico puede introducir un sesgo residual de fondo de hasta un $\pm 3.5\%$, afectando levemente el cálculo del flujo neto $I - B_{\text{local}}$.

---

## 3. Naturaleza Monocuadro vs. Superresolución Multi-frame (Drizzle)

### Estado Actual:
DeepZoom Studio opera como motor de superresolución monocuadro (SISR, *Single-Image Super-Resolution*), procesando capturas únicas renderizadas o fotográficas.

### Limitación Científica:
- En astrofotografía observacional profesional, técnicas como el algoritmo *Drizzle* (desarrollado para el Telescopio Espacial Hubble) aprovechan decenas de exposiciones con *dithering* (subpíxel jittering) para reconstruir frecuencias espaciales más allá del límite de Nyquist del sensor físico.
- Un sistema monocuadro no dispone de información observacional de muestreo temporal múltiple. Por tanto, la reconstrucción de detalles de muy alta frecuencia en las estructuras continuas descansa en los priors convolucionales de la red neuronal Remacri y en las restricciones físicas aplicadas a posteriori por los módulos astro (Hessiano, DoG, Photometric Lock).

---

## 4. Gestión de Memoria RAM en Resoluciones Extremas (8K, 12K y 16K)

### Estado Actual:
- **VRAM de GPU:** La inferencia NCNN Vulkan opera por mosaicos (*tiles*) de $400 \times 400$ px con solapamiento (*overlap*) de 10 px, requiriendo únicamente **~1.5 GB de VRAM**, lo que garantiza estabilidad absoluta en la AMD Radeon RX 5700 XT de 8 GB.
- **RAM de Sistema (CPU):** En resoluciones 4K ($2728 \times 4096 \approx 11.2$ MP), una matriz float32 requiere solo $\sim 134$ MB, ejecutándose en memoria principal sin problemas.

### Limitación Técnica en 16K:
- Para un póster vertical 16K ($9216 \times 16384 \approx 151$ Megapíxeles):
  - Una sola matriz 3D RGB en formato `np.float32` ocupa:
    $$151 \times 10^6 \times 3 \times 4 \text{ bytes} \approx 1.81 \text{ GB}$$
  - El pipeline completo, al calcular derivadas segundas Hessianas ($I_{xx}, I_{xy}, I_{yy}$), máscaras morfológicas y mapas de diferencia amplificados de 15 capas, puede requerir picos de **10 a 14 GB de memoria RAM**.
- En sistemas con 16 GB de RAM física DDR5, ejecutar el postprocesado astronómico 16K completo en una única pasada en memoria corre el riesgo de saturar la RAM y provocar *swapping* en disco.
- **Mitigación prevista para V0.2:** Procesamiento astronómico por bandas horizontales o mosaicos de postprocesado en disco con solapamiento suave, eliminando la necesidad de albergar la imagen 16K float32 entera en memoria RAM activa.
