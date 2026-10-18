# Inventario de Migración: TikTokAIStudio -> DeepZoom Studio
**Fecha:** 17 de Octubre de 2026  
**Origen:** `E:\Decalga\TikTokAIStudio\`  
**Destino:** `E:\Decalga\DeepZoomStudio\`  

---

## 1. Resumen Ejecutivo de la Auditoría

Se realizó una inspección exhaustiva de `E:\Decalga\TikTokAIStudio\` para aislar todos los componentes asociados al prototipo de superresolución previamente desarrollado. 
Se determinó que el código audiovisual base de TikTokAIStudio (narración con Kokoro TTS, transcriptor, alineadores y compositor FFmpeg) permaneció completamente desacoplado del módulo de superresolución, el cual residía exclusivamente en:
- `tools\upscaler\`
- `assets\super_resolution\`
- `docs\MANUAL_SUPERRESOLUCION_16K_VULKAN.md`

A continuación se detalla el inventario completo, su estado de verificación y las decisiones de migración.

---

## 2. Inventario Detallado de Componentes

| Archivo / Componente | Ubicación Original (TikTokAIStudio) | Función | Dependencias | ¿Migrar? | ¿Sustituible? | Licencia Conocida | Estado de Verificación |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`upscayl-bin.exe`** (v20251207) | `tools\upscaler\bin\upscayl\upscayl-bin-20251207-174704-windows\` | Motor de inferencia NCNN CLI con Vulkan, tiling y reescalado de salida. | Drivers AMD Vulkan 1.3/1.4, GPU discreta (device 0). | **SÍ** | Difícilmente (versión compilada más estable para Vulkan). | AGPLv3 (Upscayl) | **VERIFICADO** (Inferencia 16K completada en 96s). |
| **`LICENSE` & `README.md`** | `tools\upscaler\bin\upscayl\upscayl-bin-20251207-174704-windows\` | Licencia y metadatos de Upscayl-bin. | Ninguna. | **SÍ** | No (documentación legal necesaria). | AGPLv3 | **VERIFICADO** |
| **`realesrgan-ncnn-vulkan.exe`** (v0.2.5.0) | `tools\upscaler\bin\realesrgan\` | Motor NCNN alternativo de Xinntao (Tencent ARC). | `vcomp140.dll`, Vulkan. | **SÍ** (como runtime alternativo) | Sí (Upscayl-bin lo supera en opciones de resize). | BSD-3-Clause | **VERIFICADO** (Comprobado con imagen test). |
| **`vcomp140.dll` & `vcomp140d.dll`** | `tools\upscaler\bin\realesrgan\` | Runtime Microsoft OpenMP para paralelismo CPU en NCNN. | Windows VC++ Runtime. | **SÍ** (junto a RealESRGAN) | Sí (si el sistema tiene VC++ Redist instalado). | Propietaria MS Redistributable | **VERIFICADO** |
| **`remacri-4x.bin` & `.param`** | `tools\upscaler\models\` | Red neuronal ESRGAN especializada en transiciones suaves, ausencia de halos y rango dinámico estelar. | `upscayl-bin.exe` | **SÍ (CORE BASELINE)** | No (modelo ganador para astrofotografía). | CC-BY 4.0 / OpenModelDB | **VERIFICADO** (Ganador cuantitativo en horizonte y estrellas). |
| **`ultrasharp-4x.bin` & `.param`** | `tools\upscaler\models\` | Modelo 4x-UltraSharp (Kim2091) para micro-texturas y relieves. | `upscayl-bin.exe` | **SÍ** | Sí | CC-BY 4.0 | **VERIFICADO** (Evaluado en recortes). |
| **`realesrgan-x4plus.bin` & `.param`** | `tools\upscaler\models\` | Modelo base de referencia Real-ESRGAN (Xinntao). | `upscayl-bin.exe` | **SÍ** | Sí | BSD-3-Clause | **VERIFICADO** (Evaluado como control). |
| **`high-fidelity-4x.bin` & `.param`** | `tools\upscaler\models\` | Modelo para conservación de grano y detalles sutiles. | `upscayl-bin.exe` | **SÍ** | Sí | OpenModelDB | **VERIFICADO** |
| **`digital-art-4x.bin` & `.param`** | `tools\upscaler\models\` | Modelo para arte digital, anime y render sintético plano. | `upscayl-bin.exe` | **SÍ** | Sí | OpenModelDB | **VERIFICADO** |
| **`ultramix-balanced-4x.bin` & `.param`** | `tools\upscaler\models\` | Fusión equilibrada de modelos ESRGAN. | `upscayl-bin.exe` | **SÍ** | Sí | OpenModelDB | **VERIFICADO** |
| **`upscayl-standard-4x` & `lite-4x`** | `tools\upscaler\models\` | Modelos compactos y estándar de Upscayl. | `upscayl-bin.exe` | **SÍ** | Sí | OpenModelDB | **VERIFICADO** |
| **Modelos video anime (`realesr-animevideo*`)** | `tools\upscaler\bin\realesrgan\models\` | Modelos V3 para video anime (x2, x3, x4). | `realesrgan-ncnn-vulkan.exe` | **NO** (fuera de alcance de DeepZoom) | No aplica | BSD-3-Clause | **VERIFICADO** (No requerido para astro 16K). |
| **`upscaler_service.py`** | `tools\upscaler\` | Script de servicio Python con cálculo de tiling, validación y llamadas subprocess. | `PIL`, `subprocess`, `upscayl-bin`. Contiene ruta hardcodeada a TikTokAIStudio. | **SÍ (REFACTORIZAR)** | Será sustituido por `src/deepzoom/` modular. | MIT | **VERIFICADO** (Funcional, pero acoplado a TikTokAIStudio). |
| **`crop_utils.py`** | `tools\upscaler\` | Funciones para recortar porciones relativas y normalizadas. | `PIL`. | **SÍ (REFACTORIZAR)** | Se integra en `deepzoom.utils`. | MIT | **VERIFICADO** |
| **`MANUAL_SUPERRESOLUCION_16K_VULKAN.md`** | `docs\` | Documento con fundamentos de VRAM, Vulkan y modelos. | Ninguna. | **SÍ** | No (documento técnico fundacional). | Documentación propia | **VERIFICADO** |
| **`validation_report.json`** | `assets\super_resolution\` | Auditoría de dimensiones, MB y hashes de resultados. | Python/JSON. | **SÍ (a benchmarks/)** | No (referencia de verificación previa). | Datos propios | **VERIFICADO** |
| **`black_hole_source_2880x5120.jpg`** | `assets\super_resolution\originals\` | Activo fuente original (5K vertical, relación 9:16). | Ninguna. | **SÍ (a samples/astronomical/)** | No (imagen base de prueba). | Creative/Usuario | **VERIFICADO** |
| **`black_hole_base_1152x2048.png`** | `assets\super_resolution\originals\` | Imagen de referencia calibrada (Lanczos 1152x2048). | PIL. | **SÍ (a samples/astronomical/)** | No (referencia para 4x = 4608x8192). | Datos propios | **VERIFICADO** |
| **`black_hole_chat_upload_576x1024.jpg`** | `assets\super_resolution\originals\` | Muestra comprimida web (576x1024). | Ninguna. | **SÍ (a samples/astronomical/)** | No | Datos propios | **VERIFICADO** |
| **Recortes de evaluación (12 archivos PNG)** | `assets\super_resolution\crops_tests\` | Recortes 512x512 y comparativas de horizonte, disco y estrellas. | Ninguna. | **SÍ (a benchmarks/crops/)** | No (imágenes de prueba objetiva). | Datos propios | **VERIFICADO** |
| **Tiras comparativas (3 archivos PNG)** | `assets\super_resolution\crops_tests\` | Contact sheets lado a lado (Bicubic vs Remacri vs UltraSharp...). | Ninguna. | **SÍ (a docs/media/)** | No | Datos propios | **VERIFICADO** |
| **Master 8K (`blackhole_8k_remacri.png`)** | `assets\super_resolution\results_4k_8k\` | Salida 4608x8192 Remacri (28.5 MB). | Ninguna. | **SÍ (a assets/reference/)** | No (baseline para test de no-regresión). | Datos propios | **VERIFICADO** |
| **Master 8K (`blackhole_8k_ultrasharp.png`)** | `assets\super_resolution\results_4k_8k\` | Salida 4608x8192 UltraSharp (41.3 MB). | Ninguna. | **SÍ (a assets/reference/)** | No | Datos propios | **VERIFICADO** |
| **Master 12K (`blackhole_12k_remacri.png`)** | `assets\super_resolution\results_12k\` | Salida 6912x12288 Remacri (40.1 MB). | Ninguna. | **SÍ (a assets/reference/)** | No | Datos propios | **VERIFICADO** |
| **Master 16K (`blackhole_16k_remacri.png`)** | `assets\super_resolution\results_16k\` | Salida 9216x16384 Remacri (67.8 MB). | Ninguna. | **SÍ (a assets/reference/)** | No | Datos propios | **VERIFICADO** |
| **Archivo residual `results_12k.zip`** | `assets\super_resolution\` | Archivo zip duplicado creado temporalmente. | Ninguna. | **NO MIGRAR** | Sí (redundante, 42 MB). | No aplica | **VERIFICADO** |
| **FFmpeg binario** | `ffmpeg-win64-gpl` (Sistema / PATH) | Motor para extracción y renderizado de video zoompan. | Binario externo. | **NO COPIAR A REPO** (Referenciar vía PATH o config externa). | Sí (cualquier FFmpeg 6/7/8 en PATH). | GPL 8.1 | **VERIFICADO** |

---

## 3. Componentes Excluidos Explícitamente de la Migración

1. **Entorno Virtual (`.venv` en `tools\kokoro`):** No pertenece a superresolución, contiene dependencias TTS.
2. **Archivos de Audio y Voces (`assets\audio`):** No relacionados con superresolución.
3. **Scripts de Video y Subtítulos (`assemble_master_v2.py`, etc.):** Pertenecen exclusivamente al flujo TikTok.
4. **Modelos de video anime (`realesr-animevideov3`):** Innecesarios para DeepZoom Astro Engine.
5. **Archivos zip temporales (`results_12k.zip`):** Redundantes.
