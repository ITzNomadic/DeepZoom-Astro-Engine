# DeepZoom Astro Engine V0.1.3 — Sistema de Calidad FidelityGate

## 1. Misión y Fundamento

En sistemas de procesamiento científico y superresolución astronómica, cualquier alteración que degrade la calidad o introduzca artefactos no físicos debe ser detectada y bloqueada antes de su fusión.

**FidelityGate** es la compuerta de control de calidad y garantía anti-regresión introducida en DeepZoom Astro Engine V0.1.3 (`src/deepzoom/validation/fidelity_gate.py`). Su principio rector es:
> *"No introducir ningún cambio salvo que exista evidencia estadística contundente de un defecto en la salida del modelo y certeza de que la corrección propuesta es físicamente válida y superior."*

---

## 2. Arquitectura de Decisión y Criterios de Evaluación

### 2.1 Evaluación de Correcciones Estelares (`evaluate_star_correction`)

Cada parche estelar propuesto por `StarPreserve` debe superar una batería de 6 pruebas discriminantes antes de ser aplicado a la imagen:

1. **Gating de Clasificación Morfológica:**
   - Solo fuentes categorizadas estrictamente como `point_source` son admitidas.
   - Fuentes extendidas, nudos galácticos y estructuras filamentosas son rechazadas automáticamente:
     `reason = "rejected_non_point_source"`.

2. **Gating de Relación Señal-Ruido (SNR):**
   - El brillo del pico sobre el fondo debe ser representativo: $\text{SNR} \ge 3.0$.
   - Ruido de fondo o fluctuaciones estocásticas: `rejected_low_snr`.

3. **Gating de Límites Físicos de Desplazamiento:**
   - La deriva medida no puede exceder el límite físico de error de fase de superresolución ($3.0$ píxeles en HR para escala 4×).
   - Desplazamientos $> 3.0$ px corresponden a confusiones de catálogo o estrellas distintas:
     `rejected_excessive_drift`.

4. **Gating de Deriva Mínima Significativa:**
   - Si la deriva es sub-umbral ($< 0.25$ px) y no se detectan defectos de flujo ni ringing, la estrella se considera perfectamente sana:
     `rejected_healthy_star_no_drift` (preservación intacta del baseline).

5. **Gating de Integridad Energética y Contraste:**
   - El flujo del parche reconstruido no puede divergir anómalamente del baseline ($0.45 \le \Phi_{\text{rec}} / \Phi_{\text{base}} \le 2.20$).
   - Evita la amplificación espuria de artefactos: `rejected_flux_divergence`.

6. **Gating de No Inversión de Fondo:**
   - La corrección no puede hundir el fondo local por debajo del umbral del cielo oscuro:
     `rejected_background_inversion`.

### 2.2 Evaluación de Parches de Filamentos (`evaluate_filament_patch`)

Cada parche de filamento propuesto por `FilamentGuard` debe satisfacer:

1. **Umbral de Coherencia Direccional:**
   - Coherencia del campo vectorial tangencial $\ge 0.25$.
   - Estructuras isotrópicas o ruidosas: `rejected_low_coherence`.

2. **Preservación de Grosor y Ancho de Haz:**
   - El grosor medido post-reparación no puede reducirse ni ensancharse en más del 25% respecto a la estructura física:
     `rejected_unphysical_thickness_change`.

3. **Supresión de Sobre-enfoque:**
   - No se permiten incrementos de intensidad no justificados sobre la cresta central:
     `rejected_brightness_spike`.

---

## 3. Trazabilidad y Registro de Auditoría

`FidelityGate` mantiene un registro detallado en memoria accesible para benchmarking y auditoría externa:
- `evaluations`: lista completa de todas las evaluaciones realizadas, indicando `star_id`, decisión booleana `apply_correction`, motivo `decision_reason` y peso asignado.
- `summary()`: conteo acumulado de parches evaluados, aceptados, rechazados y tasa de aceptación.

```json
{
  "total_evaluations": 345,
  "accepted": 28,
  "rejected": 317,
  "acceptance_rate_pct": 8.12,
  "rejection_breakdown": {
    "rejected_healthy_star_no_drift": 272,
    "rejected_low_snr": 31,
    "rejected_non_point_source": 14
  }
}
```

En este ejemplo representativo, de 345 estrellas detectadas, **272 eran estrellas perfectamente sanas** que no requerían alteración. FidelityGate garantizó que el 91.88% de la imagen permaneciera intacto, enfocando las correcciones exclusivamente en las 28 estrellas con deriva o ringing real.
