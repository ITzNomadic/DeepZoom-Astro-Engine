# DeepZoom Astro Engine V0.1.4 — Fase I: Reconstrucción Multi-Modelo

## 1. Principio y Motivación

En las versiones anteriores (v0.1.1 - v0.1.3), todo el procesamiento astronómico dependía exclusivamente de la hipótesis generada por un único modelo convolucional: `Remacri-4x`.

Esta aproximación adolece de un fallo fundamental: si el modelo Remacri suaviza un filamento tenue o altera la fase de una estrella doble, ningún módulo posterior puede recuperar la señal perdida sin inventar información no fundada.

En V0.1.4 introducimos la **Reconstrucción Multi-Hipótesis**: generamos múltiples representaciones de alta resolución a partir de la misma imagen de entrada mediante modelos neuronales con sesgos inductivos y funciones de pérdida distintas.

---

## 2. Modelos Candidatos Soportados

| Modelo | Arquitectura / Función Objetivo | Especialidad Astronómica | Tiempo Típico 2K (GPU 0) |
|---|---|---|---|
| **Remacri-4x** | ESRGAN optimizado para texturas orgánicas suaves | **Ancla de referencia:** preserva gradientes continuos y minimiza ruido granular | ~3.5 s |
| **UltraSharp-4x** | Modelo enfocado en máxima nitidez y alta frecuencia | **Filamentos y arcos:** mayor definición en frentes de choque y bordes de discos | ~3.7 s |
| **RealESRGAN-x4plus** | Red convolucional residual con discriminador U-Net | **Texturas complejas:** separación de nudos nebulares y polvo difuso | ~3.4 s |
| **High-Fidelity-4x** | Modelo sintonizado para preservación fiel | **Estrellas y puntos:** mínima aberración cromática en fuentes puntuales | ~3.6 s |
| **Lanczos-4x** | Interpolación sinc clásica lineal | **Control analítico:** hipótesis no neuronal sin artefactos de alucinación | ~0.08 s |

---

## 3. Garantías de Consistencia

1. **Alineación Espacial Estricta:** Todas las salidas coinciden exactamente píxel a píxel con la malla de coordenadas $W_{\text{LR}} \cdot s \times H_{\text{LR}} \cdot s$.
2. **Espacio de Color Unificado:** Salidas normalizadas en RGB uint8 con rango $[0, 255]$ sin conversión destructiva a otros espacios.
3. **Inferencia Hardware Idéntica:** Mismo runtime NCNN Vulkan (`upscayl-bin.exe`), mismo dispositivo GPU (AMD Radeon RX 5700 XT), mismo tamaño de mosaico ($400 \times 400$ px con overlap de 10 px).
4. **Trazabilidad Criptográfica:** Cada candidato almacena su hash SHA-256 en el manifiesto para garantizar la reproducibilidad científica.
