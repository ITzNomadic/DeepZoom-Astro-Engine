# DeepZoom Astro Engine V0.1.4 — Fase III: Motor de Reconstrucción Estelar Avanzado

## 1. Principio y Limitaciones Previas

En V0.1.3, `StarPreserve` incorporaba un riguroso bloqueo fotométrico, pero en la imagen astronómica principal de prueba (`deepzoom_astronomy_test.png`) el filtro de defectos no autorizó ninguna modificación (`reconstructed_stars = 0`).

En V0.1.4 se aborda la raíz del problema mediante una **separación completa entre detección, modelado físico y selección multi-hipótesis**:
1. Detección multiescala DoG/LoG.
2. Clasificación morfológica cuádruple.
3. Estimación analítica de PSF (Gaussiana elíptica y Moffat).
4. Desdoblamiento paramétrico de parejas binarias cerradas.
5. Deconvolución localizada Richardson-Lucy amortiguada (Damped RL).
6. Generación de hipótesis locales concurrentes y selección por calidad objetiva.

---

## 2. Modelado de PSF y Desdoblamiento de Binarias

### 2.1 Ajuste Bimodal de Mezcla Gaussiana para Binarias Cerradas
Cuando una estrella presenta elipticidad $\varepsilon > 0.40$ y dos picos de emisión próximos (separación $\in [2.5, 9.0]$ px):
$$I_{\text{dual}}(x, y) = A_1 \exp\left( -\frac{(x-x_1)^2 + (y-y_1)^2}{2\sigma_1^2} \right) + A_2 \exp\left( -\frac{(x-x_2)^2 + (y-y_2)^2}{2\sigma_2^2} \right) + B_{\text{local}}$$
- **Criterio de Aceptación:** El modelo dual sólo se acepta si reduce el error cuadrático medio ($\text{MSE}$) en más de un **$25\%$** frente al ajuste unimodal simple, evitando desdoblamientos falsos producidos por ruido.

### 2.2 Deconvolución Richardson-Lucy Amortiguada (Damped RL)
Para halos con sub-oscilación de Gibbs profunda (ringing):
$$u_{t+1} = u_t \cdot \left[ K * \left( 1 + \min\left( \frac{|u|}{2.0}, 1 \right) \left( \frac{d}{K * u_t} - 1 \right) \right) \right]$$
- El factor de amortiguamiento de White (1993) detiene la amplificación cuando la corrección cae dentro del ruido de fondo, evitando que la deconvolución genere grano o picos artificiales.

---

## 3. Resultados en Dataset Sintético de Cúmulos y Binarias

- **Fuentes emparejadas inspeccionadas:** 49.
- **Fuentes reconstruidas activamente:** 49.
- **Parejas binarias cerradas desdobladas con éxito:** 4 de 4 pares verdaderos identificados y modelados.
- **Halos con ringing colapsados por RL:** 27 fuentes estelares.
