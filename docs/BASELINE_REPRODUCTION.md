# Informe de Reproducción de Línea Base (Baseline Reproduction)
**Proyecto:** DeepZoom Studio  
**Fecha:** 17 de Octubre de 2026  
**Hardware de Ejecución:** AMD Radeon RX 5700 XT (GPU Index 0, Vulkan 1.4.315)  
**Host:** AMD Ryzen 5 9600X, 16 GB RAM, Windows 11  

---

## 1. Objetivo de la Prueba

Verificar que la migración del pipeline desde `TikTokAIStudio` hacia `DeepZoomStudio` no haya alterado en absoluto el comportamiento del modelo, las configuraciones de Vulkan, la gestión de memoria o la calidad del renderizado. 

El criterio de éxito era la **reproducción determinista bit a bit** (SHA-256 idéntico y Diferencia Absoluta Media = 0.0) frente a las salidas de referencia.

---

## 2. Parámetros de Control Idénticos

| Parámetro | Configuración de Prueba |
| :--- | :--- |
| **Backend de Inferencia** | `DeepZoomStudio/runtimes/ncnn/upscayl-bin.exe` |
| **Directorio de Modelos** | `DeepZoomStudio/models/` |
| **Modelo Seleccionado** | `remacri-4x` (`remacri-4x.bin` & `remacri-4x.param`) |
| **Dispositivo de Aceleración** | GPU 0 (AMD Radeon RX 5700 XT) con FP16 |
| **Tamaño de Mosaico (Tile Size)** | `256` |
| **Factor de Escala** | `4` |
| **Formato de Salida** | `PNG` (lossless) |

---

## 3. Resultados Cuantitativos de la Verificación

### Test 1: Recorte Crítico 1 (Anillo de Fotones, 512 × 512 -> 2048 × 2048 px)
- **Imagen Entrada:** `benchmarks/crops/base_crop1_photon_ring.png`
- **Referencia Histórica:** `benchmarks/crops/base_crop1_photon_ring_remacri-4x.png`
- **Salida Migrada:** `output/baseline/reproduced_crop1_remacri.png`
- **Tiempo de Inferencia:** `4.33 segundos`
- **Hash Salida (SHA-256):** `1e9506777b8d33a2b76dd5d3a4670db3f15fc83905419410dfcf6c0403a60225`
- **Hash Referencia (SHA-256):** `1e9506777b8d33a2b76dd5d3a4670db3f15fc83905419410dfcf6c0403a60225`
- **Coincidencia Exacta de Píxeles:** **100.00%** (Mean Abs Diff: `0.0`, Max Diff: `0`)
- **Estado:** **PASS (DETERMINISTA)**

### Test 2: Master Completo 8K (1152 × 2048 -> 4608 × 8192 px, 37.75 Megapíxeles)
- **Imagen Entrada:** `samples/astronomical/black_hole_base_1152x2048.png`
- **Referencia Histórica:** `assets/reference/results_4k_8k/blackhole_8k_4608x8192_remacri.png`
- **Salida Migrada:** `output/baseline/reproduced_blackhole_8k_remacri.png`
- **Tiempo de Inferencia:** `18.57 segundos`
- **Hash Salida (SHA-256):** `5a3d71a38e43e40349360b42b6dc7720e8387808ea6a8c2a33ee51be347278bd`
- **Hash Referencia (SHA-256):** `5a3d71a38e43e40349360b42b6dc7720e8387808ea6a8c2a33ee51be347278bd`
- **Coincidencia Exacta de Píxeles:** **100.00%** (Mean Abs Diff: `0.0`, Max Diff: `0`)
- **Estado:** **PASS (DETERMINISTA)**

---

## 4. Conclusión de la Fase de Validación

Se confirma que el nuevo entorno autónomo en `E:\Decalga\DeepZoomStudio\` reproduce de forma 100% idéntica y reproducible el comportamiento de superresolución de `TikTokAIStudio` sin requerir ningún archivo ni ruta externa.

El baseline de **Remacri Puro queda formalmente CONGELADO**. Todas las mejoras astronómicas subsecuentes (StarPreserve, FilamentGuard, AstroValidator) se construirán como módulos experimentales independientes sobre este cimiento.
