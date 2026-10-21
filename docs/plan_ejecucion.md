# Plan de ejecución experimental actualizado

**Fecha:** 2026-10-20

Este plan reemplaza la lista informal anterior y separa claramente la secuencia principal de trabajo del experimento complementario de transferencia.

## Secuencia principal

### 1. Cerrar el protocolo de métricas con la asesora

Llevar a la reunión:

- `docs/metricas_evaluacion.md`;
- resultados actuales como ejemplo;
- curvas scratch vs. fine-tuning;
- referencias que justifican ESR, MAE, MR-STFT, LUFS y métricas de eficiencia.

Objetivo de la reunión: congelar qué métricas serán primarias, cuáles secundarias y qué configuración exacta se usará para la evaluación final.

### 2. Consolidar el pipeline común de evaluación

Estado actual:

- ✅ `scripts/evaluate_model_outputs.py`;
- ✅ validación manual de ESR sobre Starcaster50 CNN scratch;
- ✅ validación manual de ESR sobre DynaComp50 CNN scratch;
- ✅ `configs/evaluation_manifest.json`;
- ✅ `scripts/evaluate_all_existing_models.py`;
- pendiente ejecutar el batch completo;
- pendiente revisar el registro generado y congelar definitivamente las métricas después de la reunión.

Productos esperados:

- un JSON estandarizado por experimento;
- `results/evaluation/experiment_registry.csv`;
- `docs/experiment_registry.md`.

### 3. Completar LSTM scratch

Matriz faltante:

- Starcaster 25;
- Starcaster 75;
- Starcaster 100;
- Dyna Comp 25;
- Dyna Comp 75;
- Dyna Comp 100.

Total: **6 entrenamientos LSTM scratch**.

Después de cada entrenamiento:

1. seleccionar checkpoint por validation;
2. generar `test_out_bestv.wav`;
3. añadir el run canónico al manifest;
4. ejecutar el pipeline común;
5. verificar que aparezca en el registry.

### 4. Seleccionar y congelar la TCN

Primero realizar dos pilotos:

- Starcaster 50;
- Dyna Comp 50.

La arquitectura, receptive field, número de canales, dilations, loss, optimizer y demás hiperparámetros deben quedar documentados antes de lanzar el resto.

Criterio para avanzar: entrenamiento estable, salida audible válida y evaluación común funcionando.

### 5. Completar TCN scratch

Una vez congelada la TCN, entrenar:

- Starcaster 25/50/75/100;
- Dyna Comp 25/50/75/100.

Los pilotos de 50 cuentan como parte de estos ocho runs si no se cambian sus hiperparámetros después de congelar el protocolo.

### 6. Ejecutar benchmark de eficiencia

Sobre los modelos finales seleccionados:

- número de parámetros;
- tamaño del checkpoint/modelo;
- tiempo de inferencia;
- Real-Time Factor;
- memoria pico durante inferencia;
- opcionalmente MACs/FLOPs si puede medirse de forma comparable.

El benchmark debe ejecutarse bajo un mismo hardware y protocolo.

### 7. Generar resultados comparativos finales

Con la matriz completa:

- tablas por efecto/setting/arquitectura;
- ESR vs. setting;
- MAE vs. setting;
- MR-STFT vs. setting;
- LUFS error vs. setting;
- parámetros/tamaño/RTF por arquitectura;
- scatter fidelidad vs. costo computacional;
- análisis de ventajas y limitaciones por efecto.

## Matriz principal objetivo

| Efecto | Setting | CNN | LSTM | TCN |
|---|---:|---:|---:|---:|
| Starcaster | 25 | ✓ | pendiente | pendiente |
| Starcaster | 50 | ✓ | ✓ | pendiente |
| Starcaster | 75 | ✓ | pendiente | pendiente |
| Starcaster | 100 | ✓ | pendiente | pendiente |
| Dyna Comp | 25 | ✓ | pendiente | pendiente |
| Dyna Comp | 50 | ✓ | ✓ | pendiente |
| Dyna Comp | 75 | ✓ | pendiente | pendiente |
| Dyna Comp | 100 | ✓ | pendiente | pendiente |

Estado actual del núcleo: **10/24 modelos scratch**.

Faltan:

- 6 LSTM;
- 8 TCN;

total: **14 core runs**.

## Estudio complementario: transferencia de aprendizaje

Este estudio no es un paso secuencial necesario para completar la matriz principal.

Ya existen experimentos a 50 % para:

- CNN Starcaster scratch vs. Big Muff fine-tuning;
- CNN Dyna Comp scratch vs. Big Muff fine-tuning;
- LSTM Starcaster scratch vs. Wright Big Muff fine-tuning;
- LSTM Dyna Comp scratch vs. Wright Big Muff fine-tuning.

Decisión actual:

- mantenerlos como análisis complementario;
- no extender fine-tuning a 25/75/100 antes de completar CNN/LSTM/TCN scratch;
- reevaluar su expansión únicamente si sobra tiempo o la asesora lo solicita.

## Regla operativa

A partir de ahora, ningún resultado debe copiarse manualmente a tablas finales si puede ser generado automáticamente.

Flujo deseado:

```text
entrenamiento
    ↓
checkpoint seleccionado por validation
    ↓
test_out_bestv.wav
    ↓
manifest canónico
    ↓
evaluate_all_existing_models.py
    ↓
JSON + CSV + Markdown registry
    ↓
gráficas/tablas finales
```

Esto mantiene una única fuente cuantitativa de verdad y reduce errores de transcripción.
