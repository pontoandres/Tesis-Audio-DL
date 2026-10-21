# Protocolo propuesto de métricas y evaluación final

**Proyecto:** Modelado de efectos de compresión y distorsión de guitarra eléctrica mediante técnicas de aprendizaje profundo  
**Fecha de definición:** 2026-10-20  
**Estado:** propuesta para revisión con la asesora antes de congelar el protocolo final.

---

## 1. Relación con los objetivos del proyecto

La evaluación final debe responder directamente a tres objetivos:

1. **Fidelidad:** medir qué tan cercana es la señal generada por cada modelo a la salida real del pedal en los dominios temporal y frecuencial.
2. **Eficiencia computacional:** medir el costo necesario para ejecutar cada arquitectura.
3. **Compromiso fidelidad/eficiencia:** identificar si una arquitectura ofrece un mejor equilibrio global o si el resultado depende del efecto modelado.

La comparación principal debe realizarse entre **CNN, TCN y LSTM** utilizando los mismos splits de test y un protocolo de evaluación común.

**Regla importante:** las arquitecturas pueden utilizar funciones de pérdida de entrenamiento distintas, pero las métricas finales de evaluación deben calcularse externamente con el mismo código y sobre las mismas señales. Esto evita comparar directamente losses de entrenamiento que no tengan exactamente la misma definición.

---

## 2. Métricas de fidelidad recomendadas

### 2.1 Métricas primarias

Estas métricas se proponen como el conjunto principal que debe aparecer en las tablas finales.

### A. Error-to-Signal Ratio (ESR)

Métrica temporal normalizada por la energía de la señal objetivo:

[
ESR = \frac{\sum_n (y[n]-\hat{y}[n])^2}
{\sum_n y[n]^2 + \epsilon}
]

donde:

- (y[n]) es la señal real producida por el pedal;
- (hat{y}[n]) es la señal estimada por la red.

**Interpretación:** menor es mejor.  
**Ventaja:** permite comparar error respecto a la energía de cada señal y es una métrica ampliamente utilizada en modelado neuronal de amplificadores y distorsión.

Se recomienda reportar **ESR sin preénfasis como métrica común de test** para CNN, TCN y LSTM.

---

### B. Mean Absolute Error (MAE)

[
MAE = \frac{1}{N}\sum_n |y[n]-\hat{y}[n]|
]

**Interpretación:** menor es mejor.  
**Ventaja:** es sencilla, interpretable y menos sensible a errores puntuales grandes que MSE.

Se propone como segunda métrica temporal para complementar ESR.

---

### C. Multi-Resolution STFT loss / error (MR-STFT)

Compara objetivo y predicción en el dominio tiempo-frecuencia utilizando varias resoluciones de STFT.

Una implementación reproducible puede combinar:

- spectral convergence;
- distancia de magnitud logarítmica;
- varias longitudes de ventana.

**Interpretación:** menor es mejor.

Configuración inicial propuesta a 44.1 kHz:

- FFT: 512, 2048 y 8192 samples;
- hop: 128, 512 y 2048 samples;
- ventana Hann.

La configuración definitiva debe congelarse antes de generar las tablas finales.

**Motivación:** una métrica espectral es necesaria para cumplir explícitamente con el objetivo de evaluar fidelidad en el dominio frecuencial y resulta especialmente útil para detectar diferencias armónicas que un error puramente temporal puede ocultar.

---

### D. Error de loudness (LUFS error)

[
E_{LUFS}=|LUFS(y)-LUFS(\hat{y})|
]

utilizando una implementación basada en ITU-R BS.1770.

**Interpretación:** menor es mejor.

Esta métrica es particularmente relevante para el **MXR Dyna Comp**, porque permite cuantificar si el modelo reproduce adecuadamente el cambio perceptual de nivel asociado a la compresión.

Puede reportarse para ambos efectos para mantener una tabla común, pero debe destacarse su importancia específica para compresión.

---

## 3. Métricas secundarias / complementarias

Estas métricas son candidatas útiles para mostrar a la asesora, pero no es necesario adoptar todas.

### 3.1 RMSE o MSE

Mide error muestra a muestra en dominio temporal.

Es fácil de calcular, pero es parcialmente redundante con ESR y MAE. Si se quiere mantener compacta la evaluación, se recomienda **MAE + ESR** en lugar de reportar simultáneamente MAE, MSE y RMSE.

---

### 3.2 MFCC cosine distance

Calcula la distancia coseno entre representaciones MFCC de objetivo y predicción.

Ha sido utilizada en trabajos de black-box audio effect modeling como medida perceptualmente informada para tareas no lineales.

**Uso propuesto:** métrica secundaria, especialmente si la asesora solicita ampliar las medidas perceptuales.

---

### 3.3 Modulation-spectrum MSE

Compara características de modulación temporal obtenidas mediante bancos de filtros auditivos y de modulación.

Es especialmente interesante para sistemas con comportamiento temporal, pero su implementación es bastante más compleja.

**Uso propuesto:** no incluir en el conjunto mínimo; mantener como opción si se requiere una métrica adicional específicamente orientada a dinámica temporal.

---

### 3.4 Error de envolvente / dinámica para compresión

Métrica específica propuesta para el Dyna Comp:

1. obtener la envolvente de amplitud de target y predicción;
2. suavizarla con una ventana temporal definida;
3. medir MAE o RMSE entre las dos envolventes.

Esto permitiría medir directamente si la red sigue correctamente la dinámica del compresor.

**Importante:** sería una métrica diseñada para este proyecto, no una sustitución de las métricas estándar. Debe presentarse como métrica complementaria.

---

## 4. Métricas de eficiencia computacional

Estas métricas responden directamente al objetivo de caracterizar el desempeño computacional.

### 4.1 Número de parámetros entrenables

[
P = \sum_i \text{numel}(\theta_i)
]

Reportar:

- número total de parámetros;
- opcionalmente parámetros entrenables si existieran parámetros congelados.

Menor implica, en general, menor complejidad de almacenamiento y potencialmente menor costo, aunque no garantiza menor tiempo de inferencia.

---

### 4.2 Tamaño del modelo

Reportar el tamaño del checkpoint final en:

- bytes;
- kB o MB.

Debe medirse bajo un formato comparable para todos los modelos.

---

### 4.3 Tiempo de inferencia

Medir cuánto tarda cada modelo en procesar una misma cantidad de audio.

Protocolo propuesto:

- mismo computador;
- mismo archivo de entrada;
- 44.1 kHz mono;
- `torch.inference_mode()`;
- calentamiento antes de medir;
- múltiples repeticiones;
- reportar mediana y, si es posible, percentil 95.

Se recomienda utilizar un fragmento fijo de **60 s** de audio de test.

---

### 4.4 Real-Time Factor (RTF)

[
RTF = \frac{T_{procesamiento}}{T_{audio}}
]

Interpretación:

- RTF < 1: procesamiento más rápido que tiempo real;
- RTF = 1: tiempo real;
- RTF > 1: más lento que tiempo real.

También puede reportarse:

[
xRT = \frac{1}{RTF}
]

para expresar cuántas veces más rápido que tiempo real procesa el modelo.

---

### 4.5 Memoria máxima durante inferencia

Reportar memoria pico durante **inferencia**, separada de la memoria utilizada durante entrenamiento.

Para GPU puede medirse con las funciones de memoria de PyTorch. Para CPU puede utilizarse una medida de memoria del proceso.

Esta métrica es secundaria frente a parámetros, tamaño, tiempo y RTF, pero ayuda a cumplir la idea de “uso aproximado de recursos” incluida en la propuesta.

---

### 4.6 FLOPs / MACs — opcional

Puede estimarse el número de operaciones necesarias para una inferencia.

Es útil como medida teórica, pero puede ser difícil de calcular consistentemente para arquitecturas recurrentes y convolucionales con estados y secuencias de distinta longitud.

**Recomendación:** usarla solo si puede calcularse con una herramienta común para CNN, TCN y LSTM. No debe retrasar las métricas principales.

---

## 5. Conjunto mínimo recomendado para la tesis

### Fidelidad temporal
- ESR.
- MAE.

### Fidelidad espectral
- MR-STFT.

### Fidelidad perceptual/dinámica
- LUFS error.
- Opcional: MFCC cosine distance.
- Opcional específico de compresión: envelope error.

### Eficiencia
- número de parámetros;
- tamaño del modelo;
- tiempo de inferencia;
- RTF;
- memoria pico de inferencia.

Este conjunto cubre de forma directa los objetivos de la propuesta sin introducir una cantidad innecesaria de métricas redundantes.

---

## 6. Qué NO usar como única comparación

### Training loss / best validation loss

El `best_val` actualmente registrado en los modelos Wright es muy útil para:

- selección de checkpoint;
- estudiar convergencia;
- comparar scratch vs. fine-tuning cuando la función de pérdida es la misma.

Sin embargo, **no debe convertirse en la métrica principal para comparar CNN, TCN y LSTM si las arquitecturas se entrenan con funciones de pérdida diferentes**.

La evaluación arquitectónica final debe volver a calcular ESR, MAE, MR-STFT, LUFS y las métricas de eficiencia con un evaluador común.

---

## 7. Matriz experimental final

El experimento principal debe quedar balanceado:

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

Por tanto, para completar la comparación principal faltan:

- **6 entrenamientos LSTM scratch**;
- **8 entrenamientos TCN scratch**.

Los experimentos de **fine-tuning a 50 %** deben mantenerse como análisis adicional de transferencia de aprendizaje, no como sustitutos de la matriz principal.

No se recomienda repetir fine-tuning en todos los settings antes de completar CNN/LSTM/TCN scratch.

---

## 8. Estrategia de gráficas final

No es necesario generar una curva de entrenamiento para cada run en el cuerpo principal de la tesis.

### Gráficas principales de fidelidad

Para cada efecto y cada métrica:

- eje X: setting 25, 50, 75, 100;
- eje Y: métrica;
- una línea por arquitectura: CNN, LSTM, TCN.

Ejemplos:

- Starcaster — ESR vs. Distortion setting;
- Starcaster — MR-STFT vs. Distortion setting;
- Dyna Comp — ESR vs. Sensitivity;
- Dyna Comp — LUFS error vs. Sensitivity.

Esto permite comparar las tres arquitecturas y observar cómo cambia su fidelidad cuando aumenta el efecto.

### Gráficas de eficiencia

- parámetros por arquitectura;
- tamaño del modelo;
- RTF;
- tiempo de inferencia.

### Compromiso fidelidad/eficiencia

Scatter plot recomendado:

- X = RTF o número de parámetros;
- Y = ESR o MR-STFT;
- un punto por arquitectura.

Los mejores modelos estarán conceptualmente cerca de la zona de **menor error y menor costo**.

### Curvas de entrenamiento

Mantener:

- Starcaster 50 scratch vs. fine-tuning;
- Dyna Comp 50 scratch vs. fine-tuning;

como figuras del experimento adicional de transferencia.

Las curvas de todos los demás runs pueden quedar en anexos o generarse automáticamente si son necesarias.

---

## 9. Protocolo de evaluación común

Para cada modelo:

1. cargar el checkpoint seleccionado exclusivamente usando validation;
2. generar el output completo del mismo archivo de test;
3. no normalizar independientemente target y prediction;
4. verificar misma longitud, sample rate y alineación;
5. calcular todas las métricas con un único script;
6. guardar un JSON estandarizado.

Formato recomendado:

```json
{
  "effect": "starcaster",
  "setting": 50,
  "architecture": "CNN",
  "checkpoint": "model_best.json",
  "esr": 0.0,
  "mae": 0.0,
  "mrstft": 0.0,
  "lufs_error": 0.0,
  "parameters": 0,
  "model_size_bytes": 0,
  "inference_seconds": 0.0,
  "audio_seconds": 60.0,
  "rtf": 0.0,
  "peak_memory_bytes": 0
}
```

Los archivos pueden almacenarse como:

`results/evaluation/<effect>_<setting>_<architecture>.json`

y luego un script independiente puede generar automáticamente tablas y figuras.

---

## 10. Orden recomendado de ejecución

### Fase A — congelar las métricas
1. revisar este documento con la asesora;
2. decidir métricas primarias y secundarias;
3. congelar configuraciones de MR-STFT, LUFS y benchmark.

### Fase B — implementar evaluación
4. ✅ crear `scripts/evaluate_model_outputs.py`;
5. ejecutar primero sobre los modelos ya existentes;
6. verificar manualmente varios resultados.

**Estado 2026-10-20:** el evaluador común ya fue implementado. Calcula ESR, MAE, MSE, RMSE, MR-STFT y error de loudness (LUFS, mediante `pyloudnorm` cuando está instalado), sin normalizar amplitudes ni remuestrear señales. La siguiente verificación consiste en ejecutarlo sobre outputs existentes y comprobar que el ESR reproducido coincida, dentro de tolerancia numérica, con los valores ya registrados por CoreAudioML.

### Fase C — completar arquitecturas
7. entrenar los seis LSTM scratch faltantes;
8. seleccionar e implementar una TCN causal reproducible;
9. realizar primero pilotos TCN en Starcaster50 y DynaComp50;
10. si los pilotos son correctos, ejecutar 25/75/100.

### Fase D — resultados finales
11. ejecutar el evaluador común sobre todos los modelos;
12. ejecutar benchmark de eficiencia;
13. generar registro maestro CSV/JSON;
14. generar tablas y gráficas automáticamente;
15. analizar compromiso fidelidad/costo.

---

## 11. Referencias metodológicas clave

- Wright, A. & Välimäki, V. (2020). *Real-Time Guitar Amplifier Emulation with Deep Learning*. Utiliza ESR con preénfasis y analiza velocidad de procesamiento.
- Martínez Ramírez, M. A., Benetos, E. & Reiss, J. D. (2020). *Deep Learning for Black-Box Modeling of Audio Effects*. Evalúa modelos con métricas temporales y perceptualmente informadas, incluyendo MAE, MFCC cosine distance y modulation-spectrum MSE.
- Steinmetz, C. J. & Reiss, J. D. (2022). *Efficient Neural Networks for Real-Time Modeling of Analog Dynamic Range Compression*. Utiliza MAE, multi-resolution STFT y error de loudness, y compara costo computacional de TCN y LSTM.
- Steinmetz, C. J. et al. / auraloss. Implementación reproducible de pérdidas STFT y Multi-Resolution STFT para audio.

---

## 12. Preguntas concretas para llevar a la reunión

1. ¿Está de acuerdo con usar **ESR + MAE + MR-STFT + LUFS error** como conjunto principal de fidelidad?
2. ¿Conviene incluir **MFCC cosine distance** como quinta métrica o dejarla como análisis secundario?
3. ¿Le parece útil incluir una métrica específica de **envolvente/dinámica para el compresor**?
4. ¿Es suficiente caracterizar eficiencia con **parámetros + tamaño + tiempo + RTF + memoria**, dejando FLOPs como opcional?
5. ¿Está de acuerdo con utilizar las funciones de pérdida internas solo para entrenamiento/selección de checkpoint y un **evaluador externo común** para la comparación final?
6. ¿Se mantiene como conjunto definitivo de settings **25/50/75/100** y se documenta la ausencia de 0 % como desviación del plan inicial?
7. ¿Se mantiene el experimento de fine-tuning únicamente en 50 % como análisis complementario?
