# Comparación de transferencia de aprendizaje — CNN y LSTM

**Fecha:** 2026-10-16  
**Estado:** resultados experimentales preliminares para discusión con la asesora.

## 1. Objetivo

Evaluar si inicializar los modelos con pesos aprendidos sobre un **Big Muff** mejora el modelado de dos pedales físicos del dataset propio:

- **Fender Starcaster Distortion**, ajuste 50.
- **MXR Dyna Comp**, Sensitivity 50.

Se comparan pares controlados **scratch vs. fine-tuning** dentro de cada arquitectura:

- **CNN / GatedConvNet**: pesos preentrenados con el experimento Big Muff del proyecto.
- **LSTM32 / SimpleRNN**: pesos preentrenados del checkpoint Wright `muff-muff1` (LSTM, hidden size 32).

La selección del modelo se hace siempre por **mejor pérdida de validación**. El split de test se reserva para evaluación y no se usa para escoger el checkpoint.

## 2. Tabla principal de resultados

Menor es mejor para `best val`, `test loss` y `test ESR`.

| Pedal | Arquitectura | Inicialización | Epoch final | Best epoch | Best val | Test loss (best-val) | Test ESR (best-val) |
|---|---|---:|---:|---:|---:|---:|---:|
| Starcaster 50 | CNN | Scratch | 87 | 61 | 0.261490 | 0.931544 | 1.008393 |
| Starcaster 50 | CNN | Big Muff FT | 59 | 33 | **0.214686** | 0.937416 | **1.007249** |
| Dyna Comp 50 | CNN | Scratch | 27 | 1 | **1.301021** | **0.635206** | **0.854788** |
| Dyna Comp 50 | CNN | Big Muff FT | 46 | 20 | 1.372579 | 0.725555 | 0.873600 |
| Starcaster 50 | LSTM32 | Scratch | 100 | 97 | 0.229116 | **0.884839** | 1.035320 |
| Starcaster 50 | LSTM32 | Wright Big Muff FT | 53 | 27 | **0.177793** | 0.933073 | **1.027552** |
| Dyna Comp 50 | LSTM32 | Scratch | 27 | 1 | **0.766429** | 0.859836 | **0.903959** |
| Dyna Comp 50 | LSTM32 | Wright Big Muff FT | 27 | 1 | 1.269356 | **0.717394** | 1.052577 |

## 3. Cambio de scratch a fine-tuning

Los porcentajes se calculan como `(FT - scratch) / scratch × 100`. Por tanto, para las métricas de error, un valor negativo representa una reducción favorable.

| Pedal | Arquitectura | Δ best val | Δ test loss | Δ test ESR | Best epoch scratch → FT |
|---|---|---:|---:|---:|---:|
| Starcaster 50 | CNN | **-17.90%** | +0.63% | **-0.11%** | 61 → 33 |
| Dyna Comp 50 | CNN | +5.50% | +14.22% | +2.20% | 1 → 20 |
| Starcaster 50 | LSTM32 | **-22.40%** | +5.45% | **-0.75%** | 97 → 27 |
| Dyna Comp 50 | LSTM32 | +65.62% | **-16.57%** | +16.44% | 1 → 1 |

## 4. Interpretación

### 4.1 Transferencia hacia otra distorsión: Big Muff → Starcaster

La transferencia resulta **útil durante entrenamiento/validación** para las dos arquitecturas.

En la CNN, el best validation baja de 0.261490 a 0.214686, una reducción de aproximadamente **17.9%**, y el mejor checkpoint aparece en el epoch 33 en lugar del 61. El ESR de test cambia muy poco: 1.008393 → 1.007249.

En la LSTM32, el efecto sobre validación es todavía mayor: 0.229116 → 0.177793, una reducción de aproximadamente **22.4%**. Además, el mejor checkpoint aparece mucho antes, en el epoch 27 frente al 97 del entrenamiento scratch. El ESR de test también mejora ligeramente, 1.035320 → 1.027552, aunque el test loss aumenta.

Estos resultados apoyan la hipótesis de que una representación aprendida sobre un circuito de distorsión puede proporcionar una inicialización útil para otro pedal de distorsión. La evidencia es especialmente clara en **convergencia y validación**, pero no muestra todavía una mejora fuerte y consistente sobre el split de test.

### 4.2 Transferencia entre efectos distintos: Big Muff → Dyna Comp

El comportamiento es considerablemente menos favorable.

En la CNN, el fine-tuning empeora las tres métricas principales frente a scratch: best validation aumenta 5.5%, test loss 14.2% y test ESR 2.2%.

En la LSTM32 aparecen métricas contradictorias. El fine-tuning reduce el test loss de 0.859836 a 0.717394, pero empeora fuertemente el best validation (0.766429 → 1.269356) y el test ESR (0.903959 → 1.052577). Por ello no puede afirmarse que exista una ventaja consistente de transferencia para compresión.

Esto es compatible con la hipótesis experimental de que transferir desde un modelo de **distorsión** hacia un **compresor** es menos directo: el compresor introduce un comportamiento dinámico dependiente de la envolvente temporal que no coincide necesariamente con las representaciones aprendidas para una no linealidad de distorsión.

## 5. Lectura general

Los resultados actuales sugieren:

1. **Big Muff → Starcaster:** la transferencia sí aporta una inicialización útil, principalmente observable en menor pérdida de validación y convergencia más temprana.
2. **Big Muff → Dyna Comp:** no hay evidencia consistente de beneficio; en varias métricas el entrenamiento scratch es superior.
3. El beneficio de transferencia no debe juzgarse con una sola métrica. En algunos casos `test loss` y ESR cambian en direcciones distintas.
4. El checkpoint utilizado para test se mantiene definido por **mejor validación**, evitando seleccionar modelos con información del test.
5. El split de test del dataset propio ya presenta una diferencia de distribución de amplitud respecto a train/validation; por ello la brecha entre mejoras de validation y mejoras de test debe analizarse con cautela.

## 6. Comparación CNN vs. LSTM a 50%

Esta comparación es secundaria respecto al experimento de transferencia, pero deja algunas observaciones útiles:

- En **Starcaster 50**, la LSTM obtiene menor best validation que la CNN tanto scratch como fine-tuned.
- En test no existe un ganador único: las diferencias entre test loss y ESR no siempre apuntan en la misma dirección.
- En **Dyna Comp 50**, la CNN scratch obtiene mejor ESR de test (0.854788) que la LSTM scratch (0.903959).
- La comparación arquitectónica final debe incorporar también **TCN**, número de parámetros, tamaño del modelo, tiempo de inferencia y factor de tiempo real antes de concluir cuál arquitectura ofrece el mejor compromiso fidelidad/eficiencia.

## 7. Fuentes de las métricas

Resultados utilizados directamente desde el repositorio:

- `results/starcaster50_cnn100_causal128_finealign_v1-/training_stats.json`
- `results/starcaster50_cnn100_causal128_finealign_v1-/test_metrics.json`
- `results/starcaster50_cnn_ft_bigMuff_v1-/training_stats.json`
- `results/mxrdynacomp50_cnn100_causal128_v1-/training_stats.json`
- `results/mxrdynacomp50_cnn_ft_bigMuff_v1-/training_stats.json`
- `results/starcaster50_lstm32_scratch_v1-/training_stats.json`
- `results/starcaster50_lstm32_ft_wrightMuff1_v1-/training_stats.json`
- `results/mxrdynacomp50_lstm32_scratch_v1-/training_stats.json`
- `results/mxrdynacomp50_lstm32_ft_wrightMuff1_v1-/training_stats.json`

**Nota Starcaster CNN scratch:** el entrenamiento final sufrió previamente un OOM durante la evaluación integrada; por esa razón las métricas de test del checkpoint de mejor validación se toman del archivo de evaluación independiente `test_metrics.json`.

## 8. Siguientes pasos

- Generar gráficas de validation loss scratch vs. fine-tuning para Starcaster y Dyna Comp.
- Incorporar la TCN con el mismo protocolo experimental.
- Medir parámetros, tamaño en disco, latencia de inferencia y real-time factor bajo un protocolo común.
- Escuchar y documentar ejemplos de audio generados por los checkpoints de mejor validación.
- Integrar estos resultados en la sección experimental del documento de tesis después de revisión con la asesora.
