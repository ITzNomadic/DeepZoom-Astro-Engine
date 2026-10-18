# FilamentGuard V0.1.1 — Reconstrucción de Crestas y Supresión Ortogonal de Ringing

## 1. Fundamentos Físicos y Motivación

En imágenes de objetos compactos relativistas (agujeros negros supermasivos, discos de acreción, anillos de fotones y chorros relativistas), las estructuras curvilíneas delgadas portan información geométrica crítica sobre el espacio-tiempo de Kerr circundante.

Los modelos de superresolución basados en CNNs profundas introducen dos distorsiones sistemáticas sobre estas crestas:
1. **Doble borde y oscilaciones ortogonales (*Ringing Side-lobes*):** A ambos lados de una cresta brillante, la red genera valles oscuros artificiales donde la intensidad desciende de forma no física por debajo de la emisión circundante.
2. **Micro-discontinuidades a lo largo de la cresta:** El mosaico de inferencia (*tiling*) y el ruido de alta frecuencia del modelo rompen la continuidad continua del flujo luminiscente.
3. **El error de v0.1:** En la versión anterior se multiplicaba la intensidad por `(1.0 + 0.05 * mask)`, lo cual era un incremento arbitrario de brillo que no corregía ni la continuidad ni los artefactos.

---

## 2. Metodología Matemática en V0.1.1

### 2.1 Descomposición Analítica del Tensor Hessiano
Para cada píxel de la imagen se calcula la matriz Hessiana de segundas derivadas espaciales:
$$H(x, y) = \begin{pmatrix} I_{xx} & I_{xy} \\ I_{xy} & I_{yy} \end{pmatrix}$$

Mediante descomposición espectral analítica se obtienen:
- **Autovalores $\lambda_1, \lambda_2$:** Ordenados por magnitud $|\lambda_1| \ge |\lambda_2|$. Una cresta luminosa estricta presenta $\lambda_1 < 0$ con alta curvatura transversal.
- **Autovector normal transversal $\mathbf{v}_1 = (\cos\theta, \sin\theta)$:** Dirección ortogonal a la cresta, donde se manifiestan los valles de ringing.
- **Autovector tangente longitudinal $\mathbf{v}_2 = (-\sin\theta, \cos\theta)$:** Dirección a lo largo de la cresta, donde debe preservarse y estabilizarse la continuidad física.
- **Coherencia direccional:**
  $$C = \frac{(\lambda_1 - \lambda_2)^2}{(\lambda_1 + \lambda_2)^2 + \epsilon}$$

### 2.2 Supresión Ortogonal de Ringing
A una distancia subpíxel ($d \approx 1.8$ px) a lo largo de la normal $\pm \mathbf{v}_1$, FilamentGuard identifica si la señal cae en un foso de sobre-oscilación respecto al fondo circundante. De ser así, suaviza el foso restaurándolo hacia el fondo local, eliminando el halo de doble borde sin reducir la nitidez de la cresta central.

### 2.3 Suavizado Tangencial Guiado por Coherencia
A lo largo de la dirección tangente $\mathbf{v}_2$, se aplica un filtrado direccional unidimensional:
$$I_{\text{tangent}}(x, y) = 0.5 \cdot I(x, y) + 0.25 \cdot [I(x + \Delta \mathbf{v}_2) + I(x - \Delta \mathbf{v}_2)]$$
Este filtrado conecta micro-discontinuidades longitudinales causadas por el remuestreo neuronal y las costuras de teselas, sin difundir ni desenfocar el perfil transversal de la cresta.

### 2.4 Eliminación Total de Factores Arbitrarios
FilamentGuard V0.1.1 **no multiplica** el brillo por ningún factor global. Toda modificación es una corrección diferencial puramente geométrica y de eliminación de artefactos.
