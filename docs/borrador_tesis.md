# Modelado de efectos de compresión y distorsión de guitarra eléctrica mediante técnicas de aprendizaje profundo

**Borrador de documento de proyecto de grado**  
**Estudiante:** Ponto Andrés Moreno Trejos  
**Programa:** Ingeniería de Sistemas y Computación  
**Universidad:** Universidad de los Andes  
**Asesora:** Haydemar Maria Nuñez Castro  
**Año:** 2026

> Documento de trabajo para revisión. No corresponde todavía a la versión final del informe.

## Resumen provisional

Este proyecto estudia el modelado black-box de dos efectos analógicos de guitarra eléctrica, un pedal Fender Starcaster Distortion y un compresor MXR Dyna Comp, mediante tres familias de arquitecturas de aprendizaje profundo: CNN, LSTM y TCN. Se construyó un conjunto de datos propio con pares de señales de entrada y salida para cuatro configuraciones de cada pedal, correspondientes a 25 %, 50 %, 75 % y 100 % del control principal. Los modelos fueron evaluados con un protocolo común que incluye Error-to-Signal Ratio (ESR), error absoluto medio (MAE), una métrica multi-resolución basada en STFT y error absoluto de loudness medido en LUFS. También se realizó un benchmark de eficiencia computacional en CPU y GPU, midiendo número de parámetros, tamaño del estado del modelo, tiempo de inferencia, factor de tiempo real y uso de memoria CUDA.

Los resultados muestran que no existe una arquitectura que sea superior en todas las dimensiones. Para el Starcaster, la TCN presenta el menor ESR y MR-STFT promedio y además obtiene el menor RTF en GPU. Para el Dyna Comp, la LSTM obtiene el menor ESR y MR-STFT promedio, mientras la CNN conserva menor MAE y un error LUFS considerablemente menor. En CPU, la LSTM es la arquitectura más rápida, mientras que en GPU la TCN obtiene el mayor throughput. Estos resultados muestran que el compromiso entre fidelidad y eficiencia depende del efecto modelado, de la métrica utilizada y de la plataforma de ejecución.

**Palabras clave:** modelado virtual analógico, aprendizaje profundo, audio, guitarra eléctrica, CNN, LSTM, TCN, distorsión, compresión.

> **Estado de este borrador.** La matriz principal de 24 modelos, la evaluación cuantitativa y el benchmark de eficiencia ya están completos. Todavía falta definir y ejecutar la evaluación cualitativa/perceptual propuesta con la asesora y cerrar el diseño de una prueba adicional sobre material no visto. Por esa razón, las conclusiones de este documento se presentan como preliminares.

# 1. Introducción

## 1.1 Contexto y motivación

Los efectos de audio para guitarra eléctrica modifican la señal del instrumento para producir características sonoras que hacen parte directa de la interpretación. Algunos dispositivos analógicos, como los pedales de distorsión y compresión, tienen comportamientos no lineales y, en el caso de los efectos dinámicos, también dependen de la evolución temporal de la señal. Esto hace que reproducir su comportamiento con un modelo digital simple no siempre sea suficiente.

Una alternativa es el virtual analog modeling, cuyo objetivo es reproducir mediante software el comportamiento de equipos de audio físicos. Dentro de este campo, los enfoques black-box aprenden la relación entre una señal de entrada y la señal obtenida a la salida del dispositivo, sin requerir una descripción explícita de todos sus componentes electrónicos. Trabajos previos han mostrado que redes convolucionales y recurrentes pueden modelar amplificadores, circuitos de distorsión y otros sistemas de audio con resultados compatibles con aplicaciones de procesamiento en tiempo real (Wright et al., 2019; Wright et al., 2020; Martínez Ramírez et al., 2020).

Este proyecto parte de esa idea, pero la comparación no se limita a medir qué modelo produce menor error. También se busca medir cuánto cuesta ejecutar cada arquitectura. Esta segunda parte es importante porque un modelo con mayor número de parámetros no necesariamente es el más lento, y una arquitectura que funciona bien en GPU puede tener un comportamiento distinto en CPU. Por esta razón, la comparación se plantea como un compromiso entre fidelidad y eficiencia computacional.

## 1.2 Planteamiento del problema

Los pedales analógicos pueden producir transformaciones que dependen de la amplitud, del contenido espectral y del estado reciente de la señal. Aunque las técnicas de aprendizaje profundo permiten aproximar estas transformaciones directamente a partir de ejemplos, distintas arquitecturas tienen capacidades y costos diferentes. En consecuencia, escoger una arquitectura únicamente por su error de predicción o únicamente por su tamaño puede producir una conclusión incompleta.

La pregunta de investigación es:

**¿Cuál de las arquitecturas de aprendizaje profundo evaluadas ofrece el mejor equilibrio entre fidelidad de la señal y complejidad computacional para modelar efectos de compresión y distorsión de guitarra eléctrica a partir de ejemplos de entrada y salida?**

## 1.3 Objetivos

### Objetivo general

Comparar arquitecturas de aprendizaje profundo basadas en CNN, TCN y LSTM para el modelado de efectos de compresión y distorsión de guitarra eléctrica, considerando la fidelidad de la señal reproducida y la eficiencia computacional de los modelos.

### Objetivos específicos

1. Construir un conjunto de datos de señales de guitarra eléctrica compuesto por pares de audio de entrada y salida obtenidos mediante un pedal de compresión y un pedal de distorsión bajo diferentes configuraciones controladas.
2. Implementar modelos de aprendizaje profundo para el modelado de los efectos seleccionados a partir del conjunto de datos construido.
3. Evaluar la fidelidad de los modelos mediante métricas objetivas en los dominios temporal y frecuencial respecto a las señales producidas por los pedales reales.
4. Caracterizar el desempeño computacional de las arquitecturas mediante indicadores como número de parámetros, tamaño del modelo, tiempo de inferencia y factor de tiempo real.
5. Analizar el compromiso entre fidelidad sonora y costo computacional de las arquitecturas estudiadas.

## 1.4 Alcance

La comparación principal utiliza un pedal por tipo de efecto. Para distorsión se utiliza un Fender Starcaster Distortion y para compresión un MXR Dyna Comp. Se varía un único control principal de cada dispositivo y se mantienen fijos los demás controles. Aunque la propuesta inicial contemplaba cinco posiciones incluyendo 0 %, la matriz experimental finalmente se cerró con 25 %, 50 %, 75 % y 100 %. Esta decisión debe quedar explícita en la versión final porque representa un cambio respecto al plan original.

El proyecto no busca desarrollar un plugin comercial, modelar cadenas completas de efectos ni optimizar una implementación para hardware embebido específico. El objetivo es comparar, bajo condiciones controladas, tres familias de modelos representativas y documentar sus ventajas, limitaciones y requerimientos de inferencia.

# 2. Marco de referencia y trabajos relacionados

## 2.1 Modelado black-box de sistemas de audio

En el modelado black-box no se intenta reconstruir de manera explícita el circuito interno del dispositivo. El modelo aprende directamente una función que transforma la señal de entrada en una señal de salida similar a la producida por el equipo físico. Wright et al. (2019), Wright et al. (2020) y Damskägg et al. (2019) sirven como referencias principales para el uso de redes profundas en modelado neuronal de sistemas de audio.

## 2.2 Modelado de compresión

La compresión presenta una dificultad adicional porque la salida depende no solamente de la muestra actual, sino de la evolución reciente de la amplitud. Hawley et al. (2019) exploraron el perfilado de compresores con redes profundas, y Steinmetz y Reiss (2022) estudiaron redes eficientes para modelado en tiempo real de compresión de rango dinámico.

## 2.3 Arquitecturas consideradas

La comparación se concentra en tres familias. La CNN representa un modelo convolucional causal compacto. La LSTM representa una arquitectura recurrente con estado explícito. La TCN representa una alternativa basada en convoluciones causales dilatadas con un campo receptivo amplio. La intención no es demostrar que una familia sea superior en general, sino estudiar cómo cambia el compromiso entre calidad y costo cuando se modelan dos efectos con comportamientos diferentes.

# 3. Metodología

## 3.1 Diseño experimental

El proyecto se organizó como un experimento controlado con dos efectos, cuatro configuraciones por efecto y tres arquitecturas principales. Esto produce una matriz de 24 modelos entrenados desde cero. Además se conservaron cuatro experimentos de fine-tuning a 50 % como análisis complementario, pero estos no hacen parte de la matriz principal utilizada para comparar las arquitecturas.

| Efecto | Configuraciones | Arquitecturas | Modelos core |
|---|---|---|---:|
| Starcaster Distortion | 25, 50, 75, 100 % | CNN, LSTM, TCN | 12 |
| MXR Dyna Comp | 25, 50, 75, 100 % | CNN, LSTM, TCN | 12 |
| Total | | | 24 |

## 3.2 Cadena de adquisición

La adquisición se realizó con una guitarra Ibanez RG350DX y una interfaz Native Instruments Komplete Audio 6. La señal dry se reprodujo desde la salida 1 de la interfaz, pasó por el pedal físico y regresó a la entrada 3. La sesión de captura se trabajó a 44.1 kHz, con buffer de 512 muestras y Driver Error Compensation en 0 ms. En Ableton se mantuvo Warp desactivado y la señal dry sin plugins.

La misma cadena se mantuvo dentro de cada familia de capturas. En el Starcaster se varió el control Distortion. En el Dyna Comp se varió Sensitivity y el control Output se mantuvo fijo. Los targets del compresor recibieron un ajuste fijo de +6 dB de Clip Gain dentro de Ableton, aplicado de forma consistente. No se realizó normalización independiente entre input y target porque el nivel forma parte de la transformación que el modelo debe aprender.

No se utilizó una reamp box. Esta condición se mantuvo constante y se trata como una limitación experimental.

## 3.3 Preparación y alineación

Para cada efecto se construyeron pares dry/wet y se organizaron en entrenamiento, validación y prueba. Las capturas fueron convertidas a 44.1 kHz PCM16. La alineación utilizó correlación cruzada para el desfase grueso y un ajuste causal residual de aproximadamente 128 muestras, equivalentes a 2.90 ms. Este valor no se interpreta como latencia física real del pedal.

Las capturas del Starcaster mostraron inversión de polaridad respecto al dry. Se decidió no invertir los targets, ya que el modelo debe aprender la transformación observada completa.

## 3.4 Modelos

**CNN.** GatedConvNet causal basada en Automated-GuitarAmpModelling/CoreAudioML. Arquitectura representativa de 8,305 parámetros entrenables.

**LSTM.** SimpleRNN con unidad LSTM, una capa y hidden size 32. Tiene 4,513 parámetros entrenables.

**TCN.** TCN causal inspirada en TCN-300-C, con cuatro bloques, 32 canales, kernel 13, dilaciones 1, 10, 100 y 1000, campo receptivo de 13,333 muestras y 43,873 parámetros entrenables.

## 3.5 Entrenamiento y checkpoints

La selección del checkpoint se realizó exclusivamente con validación. El conjunto de prueba no se utilizó para seleccionar modelos. Para la TCN se utilizó `0.75 * ESRPre + 0.25 * DC`, preénfasis `[-0.85, 1]`, Adam con learning rate 0.005, weight decay `1e-4`, batch size 16, máximo 100 epochs y early stopping con patience 25.

## 3.6 Evaluación de fidelidad

La comparación final utiliza un evaluador común sobre las salidas de los mejores checkpoints. Las métricas principales son ESR, MAE, MR-STFT y error LUFS absoluto. Se mantienen separadas y no se construye un score único.

## 3.7 Benchmark de eficiencia

El benchmark final se ejecutó en la laptop Ryzen 7 5800HS + RTX 3060 con Python 3.11.9, PyTorch 2.3.1+cu121 y CUDA 12.1. Cada arquitectura procesó 60 s de audio sintético mono float32 a 44.1 kHz, usando chunks de 20,000 muestras, cinco warm-ups y veinte repeticiones. En CPU se usó un thread.

## 3.8 Evaluación cualitativa y generalización

> **Pendiente de definir con la asesora.** Falta cerrar el diseño del formulario perceptual y la prueba adicional sobre material no visto. No se han fijado todavía participantes, escala, fragmentos, aleatorización ni análisis.

# 4. Resultados

## 4.1 Matriz experimental

La matriz principal quedó completa con 24 modelos scratch. El manifest total contiene 28 runs al incluir cuatro experimentos complementarios de fine-tuning. El evaluador común procesó 28/28 runs sin faltantes ni errores.

## 4.2 Starcaster Distortion

| Arquitectura | ESR | MAE | MR-STFT | Error LUFS (dB) |
|---|---:|---:|---:|---:|
| CNN | 0.976334 | 0.044169 | 1.921682 | 0.390750 |
| LSTM | 1.007093 | 0.046173 | 3.252119 | 0.852250 |
| TCN | 0.935714 | 0.045729 | 1.577064 | 0.701250 |

La TCN obtiene el menor ESR y MR-STFT promedio. La CNN obtiene el menor MAE y error LUFS promedio.

## 4.3 MXR Dyna Comp

| Arquitectura | ESR | MAE | MR-STFT | Error LUFS (dB) |
|---|---:|---:|---:|---:|
| CNN | 0.967692 | 0.029390 | 2.301825 | 0.724500 |
| LSTM | 0.964000 | 0.030220 | 1.959813 | 5.215500 |
| TCN | 1.217671 | 0.035399 | 2.294639 | 4.285000 |

La LSTM obtiene el menor ESR y MR-STFT promedio. La CNN obtiene el menor MAE y una ventaja amplia en error LUFS.

## 4.4 Comportamiento de validación de Dyna Comp

Los cuatro TCN Dyna Comp tienen best epoch 1. En LSTM los mejores checkpoints aparecen entre epoch 1 y 2. El patrón se repite entre settings y entre dos familias temporales, pero todavía no se atribuye a una causa única.

## 4.5 Eficiencia computacional

| Arquitectura | Dispositivo | Params | State MB | Tiempo mediano (s) | RTF | xRT | CUDA MB |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN | CPU | 8,305 | 0.0317 | 6.463548 | 0.107726 | 9.28 | N/A |
| LSTM | CPU | 4,513 | 0.0172 | 2.359616 | 0.039327 | 25.43 | N/A |
| TCN | CPU | 43,873 | 0.1684 | 6.825922 | 0.113765 | 8.79 | N/A |
| CNN | GPU | 8,305 | 0.0317 | 0.912496 | 0.015208 | 65.75 | 42.40 |
| LSTM | GPU | 4,513 | 0.0172 | 1.110357 | 0.018506 | 54.04 | 80.68 |
| TCN | GPU | 43,873 | 0.1684 | 0.184466 | 0.003074 | 325.26 | 38.88 |

En CPU la LSTM es la más rápida. En GPU la TCN obtiene el menor RTF. Las tres arquitecturas son más rápidas que tiempo real en ambos dispositivos bajo este protocolo.

## 4.6 Fidelidad frente a eficiencia

Para los plots de trade-off, el punto de fidelidad representa el promedio descriptivo de 25/50/75/100 y la barra vertical el rango min-max observado. Los settings son condiciones distintas, no réplicas estadísticas.

En Starcaster, la TCN queda en la región más favorable de ESR vs RTF GPU y MR-STFT vs RTF GPU. En Dyna Comp no aparece la misma dominancia. La TCN es la más rápida en GPU, la LSTM favorece ESR/MR-STFT y la CNN favorece MAE/LUFS.

# 5. Discusión

## 5.1 El efecto cambia la comparación

La TCN funciona especialmente bien para Starcaster cuando se observan ESR y MR-STFT, pero esta ventaja no se traslada al compresor. En Dyna Comp, LSTM y CNN presentan ventajas diferentes y la TCN sacrifica fidelidad en varias métricas a cambio de un throughput GPU muy alto.

## 5.2 Parámetros y velocidad no significan lo mismo

La TCN tiene muchos más parámetros que CNN y LSTM, pero es la más rápida en GPU. En CPU ocurre lo contrario y la LSTM obtiene el mejor throughput. Por lo tanto, el hardware de ejecución debe formar parte de cualquier conclusión sobre eficiencia.

## 5.3 Una sola métrica no es suficiente

Los resultados de Dyna Comp muestran por qué no se debe reducir la evaluación a ESR. La LSTM obtiene ESR/MR-STFT competitivos, pero su error LUFS promedio es mucho mayor que el de la CNN. En Starcaster, la TCN gana en ESR/MR-STFT, pero CNN conserva menor MAE/LUFS.

## 5.4 Respuesta preliminar a la pregunta de investigación

No existe una arquitectura universalmente superior. Para Starcaster, la TCN ofrece el mejor compromiso observado entre ESR/MR-STFT y throughput GPU. Para Dyna Comp, la LSTM favorece fidelidad espectral y ESR promedio, la CNN favorece MAE y loudness, y la TCN favorece throughput GPU. En CPU, la LSTM es la más eficiente.

# 6. Limitaciones

1. No se utiliza reamp box.
2. Se estudia un único dispositivo por tipo de efecto.
3. La matriz final utiliza 25, 50, 75 y 100 %, sin el 0 % previsto inicialmente.
4. El benchmark mide throughput, no latencia end-to-end.
5. Falta una prueba adicional de generalización.
6. Falta la evaluación perceptual.
7. La causa del patrón de validación temprana en Dyna Comp no ha sido aislada.

# 7. Trabajo pendiente para la siguiente revisión

1. Definir el formulario de evaluación cualitativa/perceptual.
2. Definir y ejecutar la prueba adicional sobre material no visto.
3. Decidir si se añade una métrica específica de envolvente/dinámica para compresión.
4. Revisar con la asesora la justificación de cada métrica.
5. Adaptar el borrador al template final enviado por la asesora.
6. Cerrar resultados, discusión, conclusiones y póster.

# 8. Conclusiones preliminares

La matriz principal de 24 modelos ya permite comparar CNN, LSTM y TCN bajo condiciones comunes. Para Starcaster, la TCN presenta la mejor fidelidad promedio en ESR y MR-STFT y el menor RTF en GPU. Para Dyna Comp, la LSTM obtiene menor ESR/MR-STFT promedio, mientras la CNN obtiene menor MAE y error LUFS. El benchmark confirma además que la plataforma cambia el orden de eficiencia, con LSTM como la más rápida en CPU y TCN como la más rápida en GPU.

La conclusión preliminar es que no existe una arquitectura universalmente superior bajo todas las dimensiones evaluadas. La mejor elección depende del efecto, de la métrica utilizada para representar fidelidad y del hardware donde se ejecuta la inferencia. Esta conclusión todavía debe complementarse con generalización y evaluación cualitativa.

# Referencias

Damskägg, E.-P., Juvela, L., Thuillier, E., & Välimäki, V. (2019). Deep learning for tube amplifier emulation. Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 471–475.

Damskägg, E.-P., Juvela, L., & Välimäki, V. (2019). Real-time modeling of audio distortion circuits with deep learning. Proceedings of the 16th Sound & Music Computing Conference, 332–339.

Hawley, S. H., Colburn, B., & Mimilakis, S. I. (2019). SignalTrain: Profiling audio compressors with deep neural networks. arXiv:1905.11928.

Martínez Ramírez, M. A., Benetos, E., & Reiss, J. D. (2020). Deep learning for black-box modeling of audio effects. Applied Sciences, 10(2), 638. doi: 10.3390/app10020638.

Steinmetz, C. J., & Reiss, J. D. (2022). Efficient neural networks for real-time modeling of analog dynamic range compression. 152nd Convention of the Audio Engineering Society.

Wright, A., Damskägg, E.-P., & Välimäki, V. (2019). Real-time black-box modelling with recurrent neural networks. Proceedings of the 22nd International Conference on Digital Audio Effects (DAFx-19), 1–8.

Wright, A., Damskägg, E.-P., Juvela, L., & Välimäki, V. (2020). Real-time guitar amplifier emulation with deep learning. Applied Sciences, 10(3), 766. doi: 10.3390/app10030766.

Wright, A., & Välimäki, V. (2020). Perceptual loss function for neural modeling of audio systems. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 251–255. doi: 10.1109/ICASSP40776.2020.9052944.
