# StarPreserve V0.1.1 — Reconstrucción Astrofísica Local de Fuentes Puntuales

## 1. Fundamentos Físicos y Motivación

En astronomía observacional, una estrella no es una textura fotográfica genérica, sino la convolución de una emisión puntual infinitesimal con la Función de Dispersión de Punto (PSF, *Point Spread Function*) del sistema óptico o de síntesis de apertura.

Los modelos neuronales de superresolución convencionales (como Remacri o Real-ESRGAN), al haber sido entrenados con imágenes del mundo real (fotografía diurna, texturas arquitectónicas, vegetación):
1. **Generan artefactos de Gibbs y anillos de sobre-oscilación (*ringing*):** Al intentar afilar el gradiente del núcleo estelar, crean fosos oscuros concéntricos donde la intensidad cae por debajo del fondo del cielo cósmico circundante.
2. **Introducen deriva de centroide subpíxel:** Los desfases de fase de los filtros convolucionales desplazan ligeramente la coordenada baricéntrica de la estrella respecto a su posición proyectada real.
3. **Distorsionan la fotometría:** Modifican el flujo estelar integrado no linealmente, dispersando luz en halos espurios o saturando artificialmente el núcleo.
4. **Confunden nebulosas y galaxias con estrellas:** Intentar circularizar fuentes astronómicas extendidas destruye morfologías galácticas legítimas.

---

## 2. Novedades y Mejoras Clave en V0.1.1

### 2.1 Clasificación Morfológica de Fuentes
A diferencia de v0.1 (que trataba cualquier máximo local como estrella), StarPreserve V0.1.1 clasifica cada detección en tres categorías:
- **`point_source` (Estrella puntual):** Alta concentración central ($C > 2.0$), baja elipticidad ($e < 0.40$), FWHM compacto y relación señal-ruido $SNR \ge 3.5$.
- **`extended_source` (Fuente extendida):** Nudos de emisión nebular, cúmulos densos o galaxias lejanas con $e > 0.55$ o FWHM $> 16$ px. **Se protegen rigurosamente: nunca se les aplica el kernel de reconstrucción puntual.**
- **`ambiguous` (Fuente ambigua):** Detecciones en el umbral de ruido ($SNR < 3.0$). Pasan intactas al baseline sin modificación.

### 2.2 Matching Bipartito 1-a-1 Mediante Algoritmo Húngaro
- En v0.1, una correspondencia voraz (`argmin`) permitía que múltiples estrellas de entrada se asignasen a la misma estrella de salida.
- En v0.1.1, se resuelve la asignación global óptima resolviendo el problema de emparejamiento bipartito de peso mínimo mediante el Algoritmo Húngaro (`scipy.optimize.linear_sum_assignment`):
  $$D_{ij} = \sqrt{(s \cdot x_{\text{in}, i} - x_{\text{out}, j})^2 + (s \cdot y_{\text{in}, i} - y_{\text{out}, j})^2}$$
  con compuerta de exclusión estricta $D_{ij} \le d_{\max} = 1.5 \cdot s$.
- **Tratamiento de catálogos sin coincidencias:** Si $N_{\text{matched}} = 0$, los estadísticos de deriva se definen estrictamente como `None` (nunca `0.0`), el indicador de paso se fija en `False` y se reporta `NO_VALID_DISTANCE_MATCHES`.

### 2.3 Reconstrucción Astrofísica Local del Perfil PSF
En lugar de multiplicar globalmente por una máscara de oscurecimiento (`output * (1.0 - 0.3 * mask)` de v0.1), StarPreserve V0.1.1 ejecuta una reconstrucción local confinada al radio de influencia de cada fuente puntual confirmada:
1. **Estimación del fondo local ($B_{\text{loc}}$):** Mediana de intensidad en un anillo concéntrico exterior ($r \in [0.7 R, 1.0 R]$).
2. **Supresión física de fosos de ringing:** Cualquier píxel dentro del núcleo estelar ($r \le 0.8 R$) cuya intensidad caiga por debajo de $B_{\text{loc}}$ se restaura al nivel $B_{\text{loc}}$.
3. **Recentrado subpíxel:** El núcleo se alinea con la coordenada baricéntrica proyectada $(s \cdot x_{\text{in}}, s \cdot y_{\text{in}})$.
4. **Transición suave $C^1$ (Hann / Coseno Alzado):** La corrección se desvanece suavemente hacia la frontera del parche con una función coseno con derivada continua:
   $$w(r) = \frac{1}{2} \left(1 + \cos\left(\frac{\pi r}{R}\right)\right)$$
   garantizando continuidad estricta y ausencia total de costuras o artefactos de parche.
5. **Invarianza de píxeles no estelares:** Todos los píxeles fuera de las estrellas puntuales permanecen 100% idénticos bit a bit al baseline Remacri.
