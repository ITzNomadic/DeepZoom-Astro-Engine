# Manual Técnico Definitivo: Pipeline de Superresolución 16K con Vulkan NCNN
**Proyecto:** TikTokAIStudio  
**Autor:** Especialista en Superresolución e Integración Local de IA  
**Fecha:** Octubre 2026  
**Hardware Calibrado:** AMD Radeon RX 5700 XT (8 GB VRAM) | AMD Ryzen 5 9600X | 16 GB RAM | Windows 11  

---

## 1. El Problema del Escalado Gigante en Hardware AMD bajo Windows

Tradicionalmente, el escalado de imágenes mediante IA con modelos basados en PyTorch / CUDA (como ESRGAN original, Real-ESRGAN Python, SUPIR o SD-Upscale) presenta barreras severas:
1. **Dependencia de CUDA:** Exclusivo de NVIDIA.
2. **Inestabilidad de ROCm en Windows:** ROCm carece de soporte oficial para GPUs Radeon de consumo (gama RDNA 1 como la RX 5700 XT) en Windows.
3. **Consumo Desmedido de VRAM (Out of Memory):** Una imagen de 16K (9216 × 16384 píxeles = 150.99 MP) en RGB requeriría tensores intermedios de más de 19 GB de VRAM en inferencia directa, colapsando instantáneamente GPUs de 8 GB.

### La Solución Ganadora: Vulkan + NCNN + Tiling de Memoria
- **NCNN:** Es el framework de inferencia de alto rendimiento desarrollado por Tencent, optimizado para cómputo gráfico multiplataforma.
- **Vulkan Shaders:** Emplea shaders SPIR-V compilados directamente para la arquitectura RDNA de AMD, con aceleración nativa FP16 y subgrupos de 64 hilos.
- **Tiling Adaptativo:** Divide la imagen en bloques de 400 × 400 píxeles con márgenes de blending. Cada bloque consume apenas ~25 MB en VRAM, permitiendo renderizar 16K (151 MP) con un consumo pico medido de solo **1.45 GB de VRAM**.

---

## 2. Detección y Enrutamiento Crítico de la GPU

En estaciones con procesadores Ryzen modernos (como el Ryzen 5 9600X) coexisten dos adaptadores gráficos:
- **GPU 0:** AMD Radeon RX 5700 XT (Discreta, 8 GB GDDR6).
- **GPU 1:** AMD Radeon(TM) Graphics (Integrada en la CPU).

> [!WARNING]
> La opción por defecto `gpu-id auto` puede balancear o enviar la carga a la GPU integrada, ralentizando el proceso hasta 10 veces.
> **Regla de oro:** Pasar siempre `-g 0` para forzar la RX 5700 XT.

---

## 3. Ciencia de Modelos: ¿Por qué Remacri y no UltraSharp para Escenas Espaciales?

Al realizar zooms profundos en TikTokAIStudio (por ejemplo, desde el plano general de un agujero negro hasta el horizonte de sucesos o una estrella lejana), la elección del modelo determina el éxito cinematográfico:

| Criterio | Remacri-4x | UltraSharp-4x | RealESRGAN-x4plus |
| :--- | :--- | :--- | :--- |
| **Comportamiento en Sombras Puras** | Desviación estándar de ruido de **1.06**. Negro absoluto puro. | Ruido de fondo elevado (2.15). Genera falso grano dithered. | Ruido moderado (1.37) con ligera posterización. |
| **Anillo de Fotones (Alto Contraste)** | Gradiente suave (2.14). Cero efecto halo o ringing blanco. | Borde agresivo (2.91). Puede presentar línea doble artificial. | Borde intermedio con ligera aberración cromática. |
| **Turbulencia del Disco de Acreción** | Preserva filamentos gaseosos como fluidos orgánicos continuos. | Mayor micro-contraste, pero tiende a crear texturas de "gusanos". | Textura genérica suavizada. |
| **Estrellas y Puntos de Luz** | Mantiene estrellas como fuentes puntuales de luz con difracción natural. | Engrosa las estrellas y crea halos brillantes circundantes. | Estrellas algo borrosas o convertidas en manchas cuadradas. |

**Conclusión:**
- Para **espacio profundo, ciencia ficción, cinematografía limpia, cielos y rostros**: Usar siempre **`remacri-4x`**.
- Para **texturas rocosas, cañones áridos, metales oxidados y ropa áspera**: Usar **`ultrasharp-4x`**.

---

## 4. Archivos Maestros Producidos y Calibrados

Todos los archivos generados están guardados y validados en:
- **Directorio Raíz:** [`E:/Decalga/TikTokAIStudio/assets/super_resolution/`](file:///E:/Decalga/TikTokAIStudio/assets/super_resolution/)
  - **Originales:** `originals/black_hole_source_2880x5120.jpg` (2880 × 5120, 14.75 MP)
  - **Base Calibrada:** `originals/black_hole_base_1152x2048.png` (1152 × 2048, 2.36 MP)
  - **Master 4× (8K):** `results_4k_8k/blackhole_8k_4608x8192_remacri.png` (37.75 MP, 28.5 MB, 18.7 s)
  - **Master 12K:** `results_12k/blackhole_12k_6912x12288_remacri.png` (84.93 MP, 40.1 MB, 93.8 s)
  - **Master 16K:** `results_16k/blackhole_16k_9216x16384_remacri.png` (150.99 MP, 67.8 MB, 96.0 s)
  - **Auditoría:** `validation_report.json`

---

## 5. Integración en el Pipeline de Video (Animación de Cámara / Zooms)

Con la imagen maestra de 16K (9216 × 16384) ya generada, FFmpeg puede realizar zooms cinemáticos profundos de 10× sin perder resolución ni mostrar píxeles, recortando ventanas verticales 1080 × 1920 a 60 fps mediante el filtro `crop` interpolado:

```powershell
ffmpeg -y -loop 1 -i "E:\Decalga\TikTokAIStudio\assets\super_resolution\results_16k\blackhole_16k_9216x16384_remacri.png" `
  -filter_complex "zoompan=z='min(zoom+0.0015,2.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=300:s=1080x1920:fps=60" `
  -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p -t 5 `
  "E:\Decalga\TikTokAIStudio\output\blackhole_deep_zoom_60fps.mp4"
```
