# Plan de ejecución experimental actualizado

**Fecha:** 2026-11-03

Este documento refleja el estado actual del proyecto después de completar la matriz CNN/LSTM/TCN y ejecutar el evaluador común sobre todos los experimentos canónicos.

## Estado global

### Matriz principal scratch

| Efecto | Setting | CNN | LSTM | TCN |
|---|---:|---:|---:|---:|
| Starcaster | 25 | ✓ | ✓ | ✓ |
| Starcaster | 50 | ✓ | ✓ | ✓ |
| Starcaster | 75 | ✓ | ✓ | ✓ |
| Starcaster | 100 | ✓ | ✓ | ✓ |
| Dyna Comp | 25 | ✓ | ✓ | ✓ |
| Dyna Comp | 50 | ✓ | ✓ | ✓ |
| Dyna Comp | 75 | ✓ | ✓ | ✓ |
| Dyna Comp | 100 | ✓ | ✓ | ✓ |

Estado del núcleo: **24/24 modelos scratch completados**.

Además existen **4 experimentos complementarios de transferencia de aprendizaje a 50%**, por lo que el manifest contiene actualmente **28 runs canónicos**.

El evaluador común fue ejecutado sobre los **28/28 runs con 0 missing/error** y reprodujo los ESR históricos disponibles dentro de la tolerancia definida.

---

## Bloques completados

### 1. Dataset y alineación

- ✓ datasets Starcaster y Dyna Comp en 25/50/75/100;
- ✓ splits train/validation/test;
- ✓ alineación causal reproducible;
- ✓ sanity checks de nivel, duración y drift.

### 2. CNN scratch

- ✓ 8/8 runs.

### 3. LSTM32 scratch

- ✓ 8/8 runs.

### 4. TCN scratch

Se implementó una TCN causal inspirada en TCN-300-C:

- 4 bloques;
- 32 canales;
- kernel size 13;
- dilaciones 1, 10, 100, 1000;
- receptive field = 13333 muestras ≈ 302.34 ms a 44.1 kHz;
- 43,873 parámetros entrenables;
- sin FiLM/conditioning, dado que el proyecto entrena un modelo separado por setting;
- loss de entrenamiento mantenida comparable con Wright:
  - 0.75 × ESRPre;
  - 0.25 × DC.

Se ejecutaron primero pilotos a 50% y posteriormente los ocho runs scratch.

Estado: ✓ **8/8 TCN completadas**.

### 5. Evaluación común de fidelidad

Métricas comunes:

- ESR;
- MAE;
- MR-STFT;
- LUFS absolute error;
- MSE/RMSE como auxiliares.

Pipeline:

```text
checkpoint seleccionado por validation
    ↓
test_out_bestv.wav
    ↓
manifest canónico
    ↓
evaluate_all_existing_models.py
    ↓
JSON + CSV + Markdown registry
```

Estado: ✓ **28/28 runs evaluados, 0 missing/error**.

### 6. Transfer learning complementario

Ya existen experimentos a 50% para:

- CNN Starcaster scratch vs. Big Muff fine-tuning;
- CNN Dyna Comp scratch vs. Big Muff fine-tuning;
- LSTM Starcaster scratch vs. Wright Big Muff fine-tuning;
- LSTM Dyna Comp scratch vs. Wright Big Muff fine-tuning.

Se mantiene como análisis complementario. No se planea extenderlo a todos los settings salvo solicitud de la asesora o disponibilidad clara de tiempo.

---

## Próximos pasos

### 1. Generar figuras finales CNN vs. LSTM vs. TCN

Usar el registry común como fuente cuantitativa.

Figuras por efecto:

- ESR vs. setting;
- MAE vs. setting;
- MR-STFT vs. setting;
- LUFS error vs. setting.

Se prefieren líneas para la matriz principal porque los settings 25 → 50 → 75 → 100 constituyen una variable ordenada.

El análisis de transferencia a 50% se mantiene separado.

### 2. Ejecutar benchmark común de eficiencia

Medir bajo un único protocolo y hardware:

- número de parámetros;
- tamaño del checkpoint/modelo;
- tiempo de inferencia;
- Real-Time Factor (RTF);
- memoria pico durante inferencia;
- opcionalmente MACs/FLOPs si puede hacerse de forma comparable.

Pendiente definir el protocolo final de medición, incluyendo duración fija de audio, warm-up, número de repeticiones y si se reporta CPU, GPU o ambos.

### 3. Analizar formalmente fidelidad vs. costo

Con fidelidad y eficiencia completas:

- tablas por efecto/setting/arquitectura;
- comparación de medias y comportamiento por setting;
- scatter fidelidad vs. costo computacional;
- ventajas y limitaciones por arquitectura;
- diferencias entre distorsión y compresión.

### 4. Evaluación adicional de generalización

En reunión con la asesora se sugirió explorar una evaluación adicional usando material que represente claramente señal no vista durante entrenamiento.

El diseño exacto todavía está pendiente. Antes de implementarlo se debe precisar:

- qué audio constituye el conjunto adicional;
- si se trata de un nuevo test externo o una extensión del test actual;
- qué métricas se aplicarían;
- cómo evitar utilizar ese conjunto para selección de modelos.

### 5. Evaluación perceptual/cualitativa

La asesora sugirió complementar las métricas objetivas con una encuesta online basada en audios.

Idea actual, todavía no congelada:

- presentar ejemplos de target real y emulaciones;
- recoger valoración de oyentes;
- considerar participantes con y sin experiencia musical, incluyendo profesores o estudiantes de música;
- diseñar el protocolo para evitar sesgos obvios de orden, volumen o identificación de arquitectura.

La metodología exacta de la encuesta sigue pendiente.

### 6. Documento final y póster

La asesora entregó los templates/formato de:

- documento final;
- póster.

Una vez incorporados al flujo de trabajo, se debe trasladar al documento la información ya consolidada en:

- diario técnico;
- protocolo de métricas;
- registry;
- resultados comparativos;
- benchmark de eficiencia;
- evaluación perceptual si se completa.

---

## Regla operativa

A partir de este punto:

1. no copiar manualmente métricas cuando puedan generarse desde el registry;
2. cualquier nuevo experimento debe añadirse explícitamente al manifest;
3. no seleccionar modelos usando test;
4. documentar en el diario cada bloque técnico importante antes de pasar al siguiente;
5. mantener separadas las conclusiones verificadas de hipótesis todavía abiertas.

## Prioridad inmediata

1. generar figuras CNN/LSTM/TCN;
2. revisar y documentar los resultados;
3. implementar benchmark de eficiencia;
4. comenzar escritura del documento y póster en paralelo;
5. concretar con la asesora la evaluación adicional de generalización y la encuesta perceptual.
