# Tesis-Audio-DL

**Proyecto de grado — Universidad de los Andes**  
**Modelado de efectos de compresión y distorsión de guitarra eléctrica mediante técnicas de aprendizaje profundo**

Este repositorio contiene el código, los experimentos, los resultados y la documentación técnica del proyecto de grado. El objetivo es comparar arquitecturas **CNN, LSTM y TCN** para aproximar mediante aprendizaje profundo el comportamiento de pedales físicos de guitarra, evaluando tanto la **fidelidad de la señal** como la **eficiencia computacional**.

> **Estado del repositorio: 2026-10-20.** La CNN está completa para los settings 25/50/75/100 de ambos efectos; existen baselines LSTM a 50% y experimentos de transferencia a 50%. Los seis LSTM scratch restantes están actualmente en entrenamiento y la TCN todavía está pendiente de selección/congelamiento.

---

## 1. Pregunta de investigación

**¿Cuál de las arquitecturas de aprendizaje profundo evaluadas ofrece el mejor equilibrio entre fidelidad de la señal y complejidad computacional para modelar efectos de compresión y distorsión de guitarra eléctrica a partir de ejemplos de entrada y salida?**

La comparación principal se organiza sobre tres familias:

- **CNN** — GatedConvNet adaptada de Wright.
- **LSTM** — SimpleRNN con unidad LSTM; configuración actual de 32 unidades ocultas.
- **TCN** — tercera arquitectura prevista; selección final todavía pendiente.

---

## 2. Dispositivos y dataset

Se estudian dos pedales físicos:

| Efecto | Dispositivo | Control variable | Settings usados |
|---|---|---|---|
| Distorsión | Fender Starcaster Distortion | Distortion | 25, 50, 75, 100 |
| Compresión | MXR Dyna Comp | Sensitivity | 25, 50, 75, 100 |

Los controles secundarios se mantienen fijos dentro de cada serie experimental.

La adquisición se realiza mediante una **Native Instruments Komplete Audio 6** a **44.1 kHz**. Para cada configuración se construyen pares sincronizados:

~~~text
input/dry  -> señal limpia
target/wet -> salida real del pedal
~~~

Cada setting se divide en train, validation y test.

Los pares se alinean temporalmente antes del entrenamiento. El pipeline utiliza un criterio causal reproducible con un residual objetivo aproximado de **128 muestras (~2.90 ms)** después de la alineación gruesa.

### Decisiones metodológicas importantes

- No se normalizan independientemente input y target.
- No se intenta eliminar del target el comportamiento físico del pedal.
- El split de test no se utiliza para seleccionar checkpoints.
- El checkpoint representativo es siempre el de **mejor validation**.
- Las métricas finales entre arquitecturas se recalculan con un **evaluador externo común**, independientemente de la función de pérdida usada durante entrenamiento.

---

## 3. Estado experimental

### Matriz principal

| Efecto | Setting | CNN scratch | LSTM scratch | TCN scratch |
|---|---:|---:|---:|---:|
| Starcaster | 25 | ✅ | en entrenamiento | pendiente |
| Starcaster | 50 | ✅ | ✅ | pendiente |
| Starcaster | 75 | ✅ | en entrenamiento | pendiente |
| Starcaster | 100 | ✅ | en entrenamiento | pendiente |
| Dyna Comp | 25 | ✅ | en entrenamiento | pendiente |
| Dyna Comp | 50 | ✅ | ✅ | pendiente |
| Dyna Comp | 75 | ✅ | en entrenamiento | pendiente |
| Dyna Comp | 100 | ✅ | en entrenamiento | pendiente |

Antes de completar los seis LSTM en curso, la matriz central tiene **10/24 modelos scratch** terminados.

### Transfer learning — experimento complementario

También se ejecutó un estudio de transferencia a **50%**:

- CNN Starcaster: scratch vs. fine-tuning desde Big Muff.
- CNN Dyna Comp: scratch vs. fine-tuning desde Big Muff.
- LSTM Starcaster: scratch vs. fine-tuning desde Wright muff-muff1.
- LSTM Dyna Comp: scratch vs. fine-tuning desde Wright muff-muff1.

Este estudio es **complementario** y no reemplaza la matriz principal CNN/LSTM/TCN scratch.

Resultados y gráficas:

- [Comparación de transfer learning](docs/comparacion_transfer_learning.md)

---

## 4. Estructura del repositorio

~~~text
Tesis-Audio-DL/
├── configs/
│   └── evaluation_manifest.json
│
├── data/
│   ├── starcaster/
│   └── mxrdynacomp/
│
├── docs/
│   ├── diario_tesis.md
│   ├── metricas_evaluacion.md
│   ├── experiment_registry.md
│   ├── comparacion_transfer_learning.md
│   ├── plan_ejecucion.md
│   └── figures/
│
├── experiments/
│
├── external/
│   └── Automated-GuitarAmpModelling/   # submódulo Git
│
├── results/
│   ├── evaluation/
│   ├── starcaster...
│   └── mxrdynacomp...
│
└── scripts/
    ├── align_dataset.py
    ├── align_dataset_dynacomp.py
    ├── check_alignment.py
    ├── check_audio_levels.py
    ├── check_drift.py
    ├── check_drift_dynacomp.py
    ├── compare_splits.py
    ├── compare_splits_dynacomp.py
    ├── evaluate_model_outputs.py
    ├── evaluate_all_existing_models.py
    ├── plot_transfer_learning_results.py
    └── prepare50.py
~~~

---

## 5. Documentación principal

### Diario técnico

[docs/diario_tesis.md](docs/diario_tesis.md)

Es el log maestro del proyecto. Registra decisiones metodológicas, adquisición y procesamiento de datos, cambios de código, errores y fixes, comandos relevantes, experimentos, resultados, interpretación y pendientes.

No pretende ser un informe limpio: conserva también intentos fallidos y cambios de metodología para mantener trazabilidad.

### Protocolo de métricas

[docs/metricas_evaluacion.md](docs/metricas_evaluacion.md)

Define el protocolo propuesto para comparar las redes.

Métricas principales de fidelidad:

- **ESR** — Error-to-Signal Ratio.
- **MAE**.
- **MR-STFT** — error espectral multi-resolución.
- **LUFS error** — diferencia de loudness.

También se calculan MSE y RMSE como métricas auxiliares.

Métricas previstas de eficiencia:

- número de parámetros;
- tamaño del modelo;
- tiempo de inferencia;
- Real-Time Factor (RTF);
- memoria pico durante inferencia;
- FLOPs/MACs, si pueden medirse de forma comparable.

### Registro automático de experimentos

[docs/experiment_registry.md](docs/experiment_registry.md)

Tabla generada automáticamente a partir de los runs definidos en:

[configs/evaluation_manifest.json](configs/evaluation_manifest.json)

Esto evita copiar métricas manualmente y mantiene una única fuente cuantitativa reproducible.

### Plan de ejecución

[docs/plan_ejecucion.md](docs/plan_ejecucion.md)

Secuencia vigente:

1. cerrar métricas con la asesora;
2. consolidar el pipeline común de evaluación;
3. completar LSTM scratch;
4. seleccionar y congelar la TCN;
5. completar TCN scratch;
6. ejecutar benchmark de eficiencia;
7. generar resultados comparativos finales.

---

## 6. Pipeline de datos

### 6.1 Alineación

Scripts principales:

- [scripts/check_alignment.py](scripts/check_alignment.py)
- [scripts/check_drift.py](scripts/check_drift.py)
- [scripts/check_drift_dynacomp.py](scripts/check_drift_dynacomp.py)

Se utiliza correlación cruzada para estimar desfases y posteriormente se aplica un margen causal reproducible.

### 6.2 Construcción de datasets procesados

Starcaster:

~~~text
scripts/align_dataset.py
~~~

Dyna Comp:

~~~text
scripts/align_dataset_dynacomp.py
~~~

Estructura de salida esperada:

~~~text
data/processed/<effect>/<setting>/<split>/
    <name>-input.wav
    <name>-target.wav
~~~

### 6.3 Sanity checks

- [scripts/check_audio_levels.py](scripts/check_audio_levels.py)
- [scripts/compare_splits.py](scripts/compare_splits.py)
- [scripts/compare_splits_dynacomp.py](scripts/compare_splits_dynacomp.py)

Permiten verificar niveles, clipping, duración, sample rate y diferencias entre train/validation/test.

---

## 7. Modelos y código externo

La implementación base para CNN y LSTM proviene del trabajo de **Alec Wright**:

- [Automated-GuitarAmpModelling](https://github.com/Alec-Wright/Automated-GuitarAmpModelling)

Se incluye como submódulo en:

~~~text
external/Automated-GuitarAmpModelling/
~~~

El proyecto utiliza un fork propio con adaptaciones para:

- rutas externas de dataset;
- compatibilidad con el entorno PyTorch actual;
- manejo de memoria durante evaluación;
- evaluación de modelos;
- soporte práctico para los experimentos recurrentes.

CoreAudioML también se mantiene como submódulo dentro del repositorio de Wright y contiene modificaciones necesarias para los runs LSTM/cuDNN.

### Clonar correctamente

~~~bash
git clone --recurse-submodules https://github.com/pontoandres/Tesis-Audio-DL.git
cd Tesis-Audio-DL
~~~

Si ya se clonó sin submódulos:

~~~bash
git submodule update --init --recursive
~~~

---

## 8. Entorno usado

Configuración principal de entrenamiento:

- Windows;
- Python **3.11.9** dentro de .venv;
- PyTorch **2.3.1+cu121**;
- NVIDIA GeForce RTX 3060;
- NumPy;
- SciPy;
- Matplotlib;
- pyloudnorm para LUFS.

Ejemplo de creación del entorno:

~~~cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
~~~

PyTorch con CUDA 12.1:

~~~cmd
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
~~~

Dependencias auxiliares utilizadas por los scripts actuales:

~~~cmd
python -m pip install numpy scipy matplotlib pyloudnorm
~~~

> El proyecto todavía no tiene un archivo de dependencias definitivo congelado. Las versiones deberán consolidarse antes de la entrega final.

---

## 9. Evaluación común

### Evaluar un par manualmente

Script:

[scripts/evaluate_model_outputs.py](scripts/evaluate_model_outputs.py)

Ejemplo:

~~~cmd
python scripts\evaluate_model_outputs.py ^
  --target "data\processed\starcaster\50\test\starcaster50-target.wav" ^
  --prediction "results\starcaster50_cnn100_causal128_finealign_v1-\test_out_bestv.wav" ^
  --effect starcaster ^
  --setting 50 ^
  --architecture CNN ^
  --run-name starcaster50_cnn100_causal128_finealign_v1 ^
  --checkpoint model_best.json
~~~

El script calcula:

~~~text
ESR
MAE
MSE
RMSE
MR-STFT
Spectral Convergence
Log-Magnitude error
Target LUFS
Prediction LUFS
LUFS error
~~~

No normaliza ni remuestrea las señales.

### Evaluar todos los runs canónicos

~~~cmd
python scripts\evaluate_all_existing_models.py
~~~

El batch lee:

~~~text
configs/evaluation_manifest.json
~~~

y genera:

~~~text
results/evaluation/<run_id>.json
results/evaluation/experiment_registry.csv
docs/experiment_registry.md
~~~

El primer batch común evaluó **14/14 experimentos sin errores** y reprodujo todos los ESR históricos de CoreAudioML dentro de una tolerancia de 1e-5.

---

## 10. Métricas de entrenamiento vs. métricas finales

En los modelos Wright, best_val corresponde a la función de pérdida utilizada durante entrenamiento/validation:

~~~text
0.75 * ESRPre + 0.25 * DC
~~~

donde ESRPre aplica preénfasis antes del ESR.

Esta loss se utiliza para:

- selección de checkpoint;
- análisis de convergencia;
- comparación scratch vs. fine-tuning cuando la definición de loss es la misma.

**No se utiliza como única métrica para comparar CNN, LSTM y TCN.**

La comparación final entre arquitecturas se realiza recalculando las mismas métricas externas sobre el mismo split de test.

---

## 11. Convención de resultados

Cada run puede incluir archivos como:

| Archivo | Uso |
|---|---|
| model.json | estado/modelo final del run |
| model_best.json | checkpoint con mejor validation |
| training_stats.json | historial de entrenamiento y métricas |
| best_val_out.wav | output producido durante validation |
| test_out_bestv.wav | output de test del checkpoint seleccionado |
| test_out_final.wav | output de test del modelo final |
| maxmemusage.txt | registro histórico de memoria GPU del entrenamiento |
| evaluation_metrics.json | métricas del evaluador común cuando se ejecuta individualmente |

Para análisis cuantitativo se prioriza **model_best.json + test_out_bestv.wav**.

Los nombres **scratch** indican entrenamiento desde cero; **ft** indica fine-tuning desde pesos preentrenados.

---

## 12. Resultados actuales destacados

La tabla completa y actualizable está en [docs/experiment_registry.md](docs/experiment_registry.md).

Observaciones actuales:

- La transferencia **Big Muff -> Starcaster** reduce claramente la pérdida de validation para CNN y LSTM y acelera la aparición del mejor checkpoint.
- La transferencia **Big Muff -> Dyna Comp** no presenta una ventaja consistente.
- ESR, MAE, MR-STFT y LUFS pueden ordenar los modelos de manera distinta, lo que justifica una evaluación multidimensional.
- En Dyna Comp aparecen señales de posible sobreajuste temprano o diferencia de distribución entre splits en algunos runs recurrentes.
- El split de test presenta diferencias de nivel respecto a train/validation que deben considerarse al interpretar generalización.

Estas observaciones son provisionales hasta completar LSTM, TCN y el benchmark de eficiencia.

---

## 13. Próximos pasos

La prioridad inmediata es:

1. terminar los seis LSTM scratch actualmente en entrenamiento;
2. añadirlos al manifest de evaluación;
3. regenerar el registry;
4. seleccionar y validar una TCN causal sobre Starcaster50 y DynaComp50;
5. completar los ocho TCN scratch;
6. ejecutar el benchmark común de eficiencia;
7. producir tablas y gráficas finales de fidelidad vs. costo computacional.

---

## 14. Referencias de implementación

- Alec Wright — [Automated-GuitarAmpModelling](https://github.com/Alec-Wright/Automated-GuitarAmpModelling)
- Wright & Välimäki — trabajos sobre modelado neuronal de amplificadores/distorsión y ESR con preénfasis.
- Martínez Ramírez, Benetos & Reiss — Deep Learning for Black-Box Modeling of Audio Effects.
- Steinmetz & Reiss — trabajos sobre modelado eficiente de compresión dinámica y TCN.
- [auraloss](https://github.com/csteinmetz1/auraloss) — referencia para métricas y pérdidas espectrales multi-resolución.

---

## 15. Nota sobre autoría y reutilización

Las arquitecturas externas se utilizan y adaptan como implementaciones de referencia; no se presentan como desarrolladas originalmente en este proyecto.

El aporte propio incluye, entre otros:

- diseño del protocolo experimental;
- construcción del dataset físico;
- adquisición y alineación de audio;
- adaptación del código externo;
- configuración y ejecución de experimentos;
- implementación del evaluador común;
- registro automatizado de métricas;
- comparación de arquitecturas;
- análisis de fidelidad y eficiencia;
- documentación reproducible del proceso.
