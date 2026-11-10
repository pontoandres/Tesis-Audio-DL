# Modelado de efectos de compresión y distorsión de guitarra eléctrica mediante técnicas de aprendizaje profundo

**Borrador de documento de proyecto de grado**  
**Estudiante:** Ponto Andrés Moreno Trejos  
**Programa:** Ingeniería de Sistemas y Computación  
**Universidad:** Universidad de los Andes  
**Profesora asesora:** Haydemar Maria Nuñez Castro  
**Bogotá, 2026**

> Documento de trabajo para revisión. La estructura sigue el template enviado por la asesora y todavía contiene evaluaciones pendientes antes de la entrega final.

## Índice provisional

- **Resumen**
- **1. Introducción**
- **2. Contexto y planteamiento del problema**
  - 2.1 Contexto del proyecto
  - 2.2 Planteamiento del problema
  - 2.3 Pregunta de investigación
- **3. Antecedentes**
  - 3.1 Modelado black-box de sistemas de audio
  - 3.2 Modelado neuronal de distorsión y compresión
  - 3.3 Arquitecturas consideradas
- **4. Descripción de la solución propuesta**
  - 4.1 Propuesta de solución y justificación
  - 4.2 Objetivos
  - 4.3 Metodología propuesta
  - 4.4 Restricciones
- **5. Desarrollo de la solución**
  - 5.1 Diseño experimental
  - 5.2 Cadena de adquisición
  - 5.3 Preparación y alineación del conjunto de datos
  - 5.4 Implementación de los modelos
  - 5.5 Entrenamiento, herramientas y reproducibilidad
- **6. Análisis de resultados**
  - 6.1 Métodos
    - 6.1.1 Evaluación de fidelidad
    - 6.1.2 Evaluación de eficiencia computacional
    - 6.1.3 Evaluación cualitativa y generalización
  - 6.2 Validación de resultados
    - 6.2.1 Estado de la matriz experimental
    - 6.2.2 Fidelidad en Starcaster Distortion
    - 6.2.3 Fidelidad en MXR Dyna Comp
    - 6.2.4 Comportamiento de validación en Dyna Comp
    - 6.2.5 Eficiencia computacional
    - 6.2.6 Compromiso entre fidelidad y eficiencia
    - 6.2.7 Fuentes de error e incertidumbre
- **7. Conclusiones**
  - 7.1 Discusión
  - 7.2 Trabajo futuro
- **Referencias**
- **Apéndice A. Resultados completos por configuración**

# RESUMEN

Este proyecto estudia el modelado black-box de dos efectos analógicos de guitarra eléctrica, un pedal Fender Starcaster Distortion y un compresor MXR Dyna Comp, mediante tres familias de aprendizaje profundo: una red neuronal convolucional (CNN), una red recurrente Long Short-Term Memory (LSTM) y una red convolucional temporal (TCN). Se construyó un conjunto de datos propio con pares de señales de entrada y salida para cuatro configuraciones de cada pedal. Los modelos se evaluaron con un protocolo común de fidelidad que incluye ESR, MAE, MR-STFT y error de loudness en LUFS, y con un benchmark de eficiencia en CPU y GPU. La matriz principal contiene 24 modelos entrenados desde cero. Para Starcaster, la TCN obtiene el menor ESR y MR-STFT promedio, mientras la CNN obtiene menor MAE y error LUFS. Para Dyna Comp, la LSTM obtiene el menor ESR y MR-STFT promedio y la CNN presenta menor MAE y error LUFS. En CPU la LSTM obtiene el mejor throughput, mientras que en GPU la TCN es la arquitectura más rápida. Los resultados muestran que no existe una arquitectura universalmente superior y que el compromiso entre fidelidad y eficiencia depende del efecto, de la métrica y de la plataforma de ejecución.

**Palabras clave:** modelado virtual analógico, aprendizaje profundo, guitarra eléctrica, distorsión, compresión.

# 1. INTRODUCCIÓN

Los efectos de guitarra eléctrica hacen parte directa del sonido y de la interpretación del instrumento. Pedales como la distorsión y la compresión modifican la forma de onda, el contenido espectral y la dinámica de la señal. Aunque estas transformaciones pueden realizarse mediante procesamiento digital, reproducir con precisión el comportamiento de un dispositivo analógico no siempre es sencillo, especialmente cuando existen no linealidades o dependencias temporales.

El modelado virtual analógico busca reproducir mediante software el comportamiento de equipos físicos de audio. Dentro de este campo, los métodos black-box permiten aprender una transformación directamente a partir de ejemplos de entrada y salida sin reconstruir de forma explícita el circuito electrónico. Este enfoque resulta atractivo porque puede utilizarse con dispositivos cuya respuesta completa es difícil de describir mediante una formulación analítica y porque permite comparar diferentes modelos bajo un mismo conjunto de datos.

El problema abordado en este proyecto no consiste únicamente en obtener una señal parecida a la producida por un pedal real. Para que un modelo tenga interés práctico también debe poder ejecutarse con un costo computacional razonable. Una arquitectura con un error menor puede requerir más memoria o más tiempo de inferencia, y un modelo con más parámetros no necesariamente es más lento en todas las plataformas. Por esta razón, el proyecto estudia de manera conjunta la fidelidad y la eficiencia computacional.

Esta comparación tiene una relevancia práctica clara. Un modelo digital suficientemente fiel puede permitir reproducir el carácter de un dispositivo físico sin tener el pedal conectado durante cada uso. Sin embargo, en una aplicación musical el procesamiento debe producirse mientras se reproduce o interpreta audio. Por eso no basta con que el modelo funcione fuera de línea. También interesa determinar si puede procesar la señal más rápido que su duración real y cómo cambia ese comportamiento entre CPU y GPU.

Una solución adecuada para este problema debe cumplir varias condiciones. Debe aprender la transformación a partir de ejemplos reales del dispositivo, debe evaluarse sobre audio que no se utilizó para ajustar los pesos y debe compararse con criterios que no favorezcan de antemano a una arquitectura. Además, debe conservar suficiente información sobre el procedimiento experimental para que los resultados puedan reproducirse y para que las diferencias observadas puedan atribuirse a los modelos y no a cambios en la adquisición o en las métricas.

La solución desarrollada consiste en construir un conjunto de datos pareado a partir de dos pedales físicos, entrenar tres familias de modelos sobre las mismas divisiones de datos y evaluarlas con un protocolo común. Se estudian una CNN causal, una LSTM y una TCN causal. Cada familia se entrena de forma independiente para cuatro configuraciones del control principal de cada pedal. Posteriormente se comparan las predicciones contra las señales reales y se mide el costo de inferencia en CPU y GPU.

El proyecto produce tres resultados principales. Primero, una matriz experimental completa de modelos comparables bajo las mismas condiciones. Segundo, un evaluador común que separa varias dimensiones de fidelidad en lugar de reducir toda la comparación a una única loss de entrenamiento. Tercero, un benchmark de inferencia que permite relacionar la calidad obtenida con el costo real de ejecutar cada arquitectura. Estos tres elementos forman la base para responder la pregunta de investigación.

El objetivo general es comparar las tres arquitecturas para determinar qué compromiso ofrecen entre fidelidad de la señal y eficiencia computacional al modelar distorsión y compresión de guitarra eléctrica. Los objetivos específicos incluyen construir el conjunto de datos, implementar los modelos, evaluar la fidelidad, caracterizar el desempeño computacional y analizar el compromiso entre ambas dimensiones.

El documento se organiza de la siguiente forma. La Sección 2 presenta el contexto y formula el problema de investigación. La Sección 3 resume los antecedentes principales y las familias de modelos estudiadas. La Sección 4 describe la solución propuesta, sus objetivos, metodología y restricciones. La Sección 5 documenta la construcción de la solución y las decisiones de implementación. La Sección 6 presenta el protocolo de evaluación y los resultados obtenidos. Finalmente, la Sección 7 discute los hallazgos, las limitaciones y el trabajo pendiente o futuro.

# 2. CONTEXTO Y PLANTEAMIENTO DEL PROBLEMA

## 2.1 Contexto del proyecto

En una cadena tradicional de guitarra eléctrica, un pedal recibe una señal del instrumento y aplica una transformación antes de enviarla a un amplificador, una interfaz u otro efecto. Los pedales analógicos producen estas transformaciones mediante circuitos físicos. En el caso de la distorsión, la modificación incluye no linealidades fuertes que alteran la forma de onda y generan nuevo contenido espectral. En un compresor, la ganancia aplicada depende de la amplitud y de la evolución temporal de la señal.

El modelado digital de estos dispositivos permite reproducir su comportamiento sin disponer permanentemente del hardware original. Esta posibilidad tiene aplicaciones en producción musical, efectos virtuales y sistemas de audio en tiempo real. Sin embargo, el modelo debe conservar suficiente fidelidad y, al mismo tiempo, poder procesar audio con una velocidad compatible con el uso práctico.

El proyecto utiliza dos dispositivos disponibles para la adquisición experimental: un Fender Starcaster Distortion y un MXR Dyna Comp. Se estudian cuatro configuraciones del control principal de cada pedal, correspondientes a 25 %, 50 %, 75 % y 100 %. La comparación se concentra en CNN, LSTM y TCN porque representan tres formas diferentes de procesar contexto temporal y porque cuentan con antecedentes directos en modelado neuronal de audio.

## 2.2 Planteamiento del problema

Las arquitecturas de aprendizaje profundo no presentan el mismo comportamiento frente a todos los efectos de audio. Una red recurrente puede representar dependencias temporales mediante un estado interno, mientras una red convolucional puede explotar paralelismo y una TCN puede ampliar su campo receptivo mediante dilataciones. Estas diferencias afectan tanto el error de predicción como el costo de ejecución.

Evaluar únicamente una loss de entrenamiento no permite determinar qué modelo es más adecuado. Dos modelos pueden ordenar de manera distinta según se observe el error muestra a muestra, el contenido tiempo-frecuencia o el loudness. De la misma forma, el número de parámetros no determina por sí solo el tiempo de inferencia. La comparación requiere entonces un protocolo común que mida varias dimensiones de fidelidad y varias dimensiones de eficiencia.

El problema de investigación consiste en determinar cómo cambia ese compromiso cuando se modelan dos efectos con comportamientos diferentes bajo condiciones experimentales controladas. El propósito no es identificar una arquitectura universalmente superior, sino establecer qué ventajas y limitaciones aparecen para cada efecto y plataforma de ejecución.

## 2.3 Pregunta de investigación

**¿Cuál de las arquitecturas de aprendizaje profundo evaluadas ofrece el mejor equilibrio entre fidelidad de la señal y complejidad computacional para modelar efectos de compresión y distorsión de guitarra eléctrica a partir de ejemplos de entrada y salida?**

# 3. ANTECEDENTES

## 3.1 Modelado black-box de sistemas de audio

En el modelado black-box no se reconstruye de forma explícita el circuito interno de un dispositivo. En cambio, el modelo aprende una función que transforma una señal de entrada en una señal de salida similar a la producida por el sistema físico. Este enfoque es especialmente útil cuando la respuesta del sistema incluye no linealidades o memoria y no se dispone de un modelo analítico sencillo.

Wright et al. (2019) estudiaron el modelado black-box en tiempo real mediante redes recurrentes. Posteriormente, Wright et al. (2020) aplicaron técnicas de aprendizaje profundo al modelado de amplificadores de guitarra. Damskägg et al. (2019) presentaron resultados para emulación de amplificadores y circuitos de distorsión. Estos trabajos muestran que modelos neuronales compactos pueden reproducir sistemas de audio no lineales y sirven como referencia directa para la implementación utilizada en este proyecto.

## 3.2 Modelado neuronal de distorsión y compresión

La distorsión modifica de forma no lineal la amplitud de la señal y genera componentes espectrales que no estaban presentes en la entrada. Por esta razón, una evaluación adecuada debe observar no solamente la diferencia temporal entre formas de onda, sino también el contenido espectral generado por el modelo.

La compresión introduce una dificultad diferente. La ganancia aplicada depende de la evolución reciente de la amplitud y, por lo tanto, el sistema presenta memoria. Hawley et al. (2019) exploraron el perfilado de compresores mediante redes profundas. Steinmetz y Reiss (2022) estudiaron redes eficientes para modelado en tiempo real de compresión de rango dinámico. Estos trabajos motivan el uso de modelos capaces de incorporar contexto temporal y la necesidad de evaluar su costo de inferencia.

Martínez Ramírez et al. (2020) presentan un marco general de aprendizaje profundo para modelado black-box de efectos de audio. En conjunto, estos antecedentes muestran que no existe una única arquitectura obligatoria para este problema y justifican realizar una comparación directa bajo un mismo conjunto de datos y un mismo evaluador.

## 3.3 Arquitecturas consideradas

La **CNN** utilizada representa una arquitectura convolucional causal compacta. Las convoluciones permiten procesar patrones locales de la señal y, al ser causales, evitan utilizar muestras futuras durante la inferencia.

La **LSTM** es una arquitectura recurrente que mantiene un estado interno. Esta característica permite representar dependencias temporales sin utilizar una ventana fija de entrada, pero su ejecución es inherentemente secuencial en parte del cálculo.

La **TCN** utiliza convoluciones causales dilatadas. Las dilataciones permiten construir un campo receptivo amplio sin incrementar de forma proporcional el número de capas. Este tipo de arquitectura puede procesar secuencias de forma altamente paralela en GPU.

# 4. DESCRIPCIÓN DE LA SOLUCIÓN PROPUESTA

## 4.1 Propuesta de solución y justificación

La solución propuesta consiste en construir una cadena experimental reproducible que permita comparar CNN, LSTM y TCN utilizando exactamente los mismos efectos, configuraciones y divisiones de datos. La entrada del sistema es una señal de guitarra limpia y el objetivo de cada modelo es predecir la señal producida por el pedal físico correspondiente.

Para cada efecto se capturan pares sincronizados de señal dry, es decir, la señal limpia antes del pedal, y señal wet, que corresponde a la salida procesada por el dispositivo. El conjunto de datos se divide en entrenamiento, validación y prueba. Cada arquitectura se entrena de manera independiente para 25 %, 50 %, 75 % y 100 % del control principal del pedal. Esta decisión reduce variables adicionales durante la comparación, ya que cada modelo aprende una única configuración física.

La justificación de la solución es que una comparación entre arquitecturas solo es útil si las condiciones de adquisición, los datos de prueba y el evaluador son comunes. Por esta razón, la loss interna de entrenamiento se utiliza para seleccionar checkpoints, pero la comparación final se recalcula mediante un evaluador externo idéntico para todas las arquitecturas.

Los resultados esperados de la solución son: un conjunto de datos pareado reproducible, modelos entrenados para los dos efectos y cuatro configuraciones, métricas objetivas de fidelidad calculadas bajo un protocolo común y una caracterización de eficiencia que permita determinar si los modelos pueden operar más rápido que tiempo real. También se espera identificar si el balance entre fidelidad y costo cambia según el efecto o la plataforma de ejecución.

## 4.2 Objetivos

### Objetivo general

Comparar arquitecturas de aprendizaje profundo basadas en CNN, TCN y LSTM para el modelado de efectos de compresión y distorsión de guitarra eléctrica, considerando la fidelidad de la señal reproducida y la eficiencia computacional de los modelos.

### Objetivos específicos

1. Construir un conjunto de datos de señales de guitarra eléctrica compuesto por pares de audio de entrada y salida obtenidos mediante un pedal de compresión y un pedal de distorsión bajo diferentes configuraciones controladas.
2. Implementar modelos de aprendizaje profundo para el modelado de los efectos seleccionados a partir del conjunto de datos construido.
3. Evaluar la fidelidad de los modelos mediante métricas objetivas en los dominios temporal y frecuencial respecto a las señales producidas por los pedales reales.
4. Caracterizar el desempeño computacional de las arquitecturas mediante indicadores como número de parámetros, tamaño del modelo, tiempo de inferencia y factor de tiempo real.
5. Analizar el compromiso entre fidelidad sonora y costo computacional de las arquitecturas estudiadas.

## 4.3 Metodología propuesta

La metodología se divide en cinco etapas. Primero se realiza la adquisición de audio utilizando una señal dry reproducida a través de los pedales físicos y se registra la salida wet. Segundo, las señales se convierten a un formato común y se alinean temporalmente. Tercero, se entrenan CNN, LSTM y TCN para cada configuración. Cuarto, se selecciona un checkpoint utilizando únicamente validación y se calcula la fidelidad sobre el conjunto de prueba con un evaluador común. Quinto, se ejecuta un benchmark de inferencia en CPU y GPU y se cruza el costo computacional con los resultados de fidelidad.

La matriz principal contiene dos efectos, cuatro configuraciones por efecto y tres arquitecturas, para un total de 24 modelos entrenados desde cero. Los experimentos de fine-tuning a 50 % se conservan como análisis complementario y no modifican la comparación principal.

La evaluación objetiva se complementará con una prueba sobre material claramente no visto y con una evaluación perceptual. Estas dos actividades continúan pendientes de diseño definitivo con la asesora y no se presentan como resultados completados en este borrador.

## 4.4 Restricciones

**Económicas y de infraestructura.** El proyecto se desarrolló principalmente con equipos ya disponibles, incluyendo la guitarra, los pedales, la interfaz de audio, un computador de escritorio y una laptop con GPU. No se utilizó infraestructura de cómputo en la nube ni una reamp box dedicada. Esta última decisión reduce costos, pero introduce una limitación en la equivalencia eléctrica de la cadena de adquisición.

**Ambientales.** El proyecto no requiere fabricación de hardware ni materiales adicionales. El consumo de energía asociado a entrenamiento e inferencia existe, pero no fue medido de forma independiente y no constituye una variable comparada entre modelos.

**Sociales y éticas.** La parte cuantitativa utiliza grabaciones producidas dentro del proyecto y no contiene información personal de terceros. La evaluación perceptual prevista sí involucra participantes humanos. Antes de ejecutarla se debe definir consentimiento, manejo de respuestas, anonimización y nivel de exposición sonora. Esta evaluación todavía no se ha realizado.

**Legales y de propiedad intelectual.** Parte de la implementación se apoya en código abierto de Automated-GuitarAmpModelling y CoreAudioML. Las modificaciones propias y el uso de estos repositorios deben conservar atribución y respetar sus condiciones de licencia. Las referencias académicas utilizadas también deben mantenerse identificadas de forma explícita.

**Salud y seguridad.** La adquisición se realizó a niveles de trabajo normales de audio. Para una evaluación perceptual se deberá indicar a los participantes que ajusten el volumen a un nivel cómodo y evitar estímulos innecesariamente fuertes.

**Políticas.** No se identificaron restricciones políticas que afecten de forma material el desarrollo técnico del proyecto.

**Mantenimiento y reproducibilidad.** El proyecto depende de versiones concretas de Python, PyTorch, CUDA y librerías de audio. El repositorio conserva scripts, configuraciones y documentación para reducir el riesgo de que futuras actualizaciones de dependencias impidan reproducir los experimentos.

# 5. DESARROLLO DE LA SOLUCIÓN

## 5.1 Diseño experimental

El experimento principal utiliza dos efectos, cuatro configuraciones y tres arquitecturas. En el documento se utiliza el término setting para referirse a una posición concreta del control principal del pedal. Se entrena un modelo separado para cada combinación de efecto, setting y arquitectura.

| Efecto | Configuraciones | Arquitecturas | Modelos principales |
|---|---|---|---:|
| Fender Starcaster Distortion | 25, 50, 75, 100 % | CNN, LSTM, TCN | 12 |
| MXR Dyna Comp | 25, 50, 75, 100 % | CNN, LSTM, TCN | 12 |
| **Total** | | | **24** |

Además se conservaron cuatro experimentos de fine-tuning a 50 % como análisis complementario. La comparación principal del proyecto utiliza los 24 modelos entrenados desde cero.

## 5.2 Cadena de adquisición

La adquisición se realizó con una guitarra Ibanez RG350DX y una interfaz Native Instruments Komplete Audio 6. La señal dry se reprodujo desde la salida 1 de la interfaz, pasó por el pedal físico y regresó a la entrada 3. Las sesiones se trabajaron a 44.1 kHz, con buffer de 512 muestras y Driver Error Compensation en 0 ms. En Ableton Live se mantuvo Warp desactivado y la señal dry sin plugins.

En el Starcaster se varió el control Distortion. En el Dyna Comp se varió Sensitivity y el control Output se mantuvo fijo en su valor definido para la serie. Los targets del compresor recibieron un ajuste fijo de +6 dB de Clip Gain en Ableton. Este ajuste se aplicó de forma consistente a todas las capturas del efecto y no se realizó normalización independiente entre input y target.

No se utilizó una reamp box. La salida de línea de la interfaz se conectó directamente a los pedales. Esta condición se mantuvo constante durante las capturas para evitar que se convirtiera en una variable adicional entre modelos.

## 5.3 Preparación y alineación del conjunto de datos

Para cada efecto se construyeron pares de señales dry y wet y se organizaron en divisiones de entrenamiento, validación y prueba. Las capturas se convirtieron a 44.1 kHz y PCM16. Los archivos originales se conservaron separados de los archivos procesados utilizados durante el entrenamiento.

La alineación se realizó en dos etapas. Primero se estimó un desfase grueso mediante correlación cruzada. Después se aplicó un margen causal residual de aproximadamente 128 muestras, equivalente a 2.90 ms a 44.1 kHz. Este valor no representa una medición de la latencia física del pedal, sino una decisión reproducible del pipeline para garantizar contexto causal.

Las capturas de Starcaster presentan inversión de polaridad respecto a la señal dry. Se decidió conservar esta transformación en los targets en lugar de invertirla artificialmente. En Dyna Comp, la correlación local no presentó un signo estable en todas las ventanas, por lo que la estimación temporal se realizó utilizando la magnitud de la correlación dentro de una región restringida.

## 5.4 Implementación de los modelos

**CNN.** Se utilizó una GatedConvNet causal basada en Automated-GuitarAmpModelling/CoreAudioML de Wright. La arquitectura representativa utilizada en el benchmark tiene 8,305 parámetros entrenables.

**LSTM.** Se utilizó una SimpleRNN con unidad LSTM, una capa recurrente y hidden size 32. La arquitectura tiene 4,513 parámetros entrenables. Durante la adaptación fue necesario asegurar que los tensores recurrentes fueran contiguos para evitar errores de cuDNN en GPU.

**TCN.** Se implementó una TCN causal inspirada en TCN-300-C de Steinmetz y Reiss. La red utiliza cuatro bloques, 32 canales, kernel 13 y dilaciones 1, 10, 100 y 1000. El campo receptivo es de 13,333 muestras, aproximadamente 302.34 ms a 44.1 kHz, y la arquitectura tiene 43,873 parámetros entrenables. Se entrenó una red independiente para cada configuración del pedal. No se entrenó una única red condicionada por el valor del setting mediante técnicas como FiLM.

## 5.5 Entrenamiento, herramientas y reproducibilidad

La selección del checkpoint, es decir, el estado guardado del modelo que se conserva como candidato final, se realizó exclusivamente con el conjunto de validación. El conjunto de prueba se utilizó después de escoger el mejor checkpoint y no intervino en esa selección. Esto evita seleccionar modelos utilizando directamente la información con la que posteriormente se reporta su desempeño.

Para la TCN se utilizó una función de pérdida compuesta por 0.75 ESRPre y 0.25 DC, con preénfasis [-0.85, 1], Adam con learning rate inicial de 0.005, weight decay de 1e-4, batch size 16, máximo 100 epochs y early stopping con patience 25. CNN y LSTM conservaron el pipeline de Wright/CoreAudioML y su comparación final se homogeneizó mediante el evaluador externo común.

El desarrollo se realizó principalmente en Python 3.11.9 y PyTorch. Para entrenamiento se utilizó una laptop con AMD Ryzen 7 5800HS, 16 GB de RAM y NVIDIA RTX 3060 Laptop GPU. El repositorio del proyecto conserva el código de preparación, entrenamiento, evaluación, generación de figuras y documentación técnica. Automated-GuitarAmpModelling se mantiene como submódulo y CoreAudioML se utiliza mediante un fork con la adaptación necesaria para los experimentos recurrentes en cuDNN.

# 6. ANÁLISIS DE RESULTADOS

## 6.1 Métodos

### 6.1.1 Evaluación de fidelidad

La comparación de fidelidad no utiliza directamente la loss interna de entrenamiento. Después de seleccionar el checkpoint de cada run, todas las predicciones se procesan mediante un evaluador común sobre el mismo split de prueba, con la misma frecuencia de muestreo, alineación y longitud. Target y predicción no se normalizan de forma independiente.

Se utilizan cuatro métricas principales, todas interpretadas como menor es mejor:

1. **ESR, Error-to-Signal Ratio.** Compara la energía del error contra la energía de la señal objetivo. Se utiliza porque permite comparar error de forma normalizada respecto al nivel de la señal.
2. **MAE, error absoluto medio.** Mide la diferencia promedio muestra a muestra y permite observar error de amplitud sin elevar al cuadrado las diferencias.
3. **MR-STFT.** Compara la representación tiempo-frecuencia a varias resoluciones. Se utilizan FFT de 512, 2048 y 8192 muestras, con hops de 128, 512 y 2048. Esta métrica complementa las métricas temporales porque la distorsión modifica directamente el contenido espectral.
4. **Error LUFS absoluto.** Mide la diferencia de loudness global entre target y predicción. Se incluye porque dos señales pueden obtener un error temporal o espectral similar y aun así presentar una diferencia perceptible de nivel global.

No se construye un score único. Cada métrica representa un aspecto distinto de la señal y combinarlas mediante pesos arbitrarios podría ocultar diferencias relevantes entre arquitecturas.

### 6.1.2 Evaluación de eficiencia computacional

El benchmark final, entendido como una prueba controlada de rendimiento, se ejecutó en la laptop Ryzen 7 5800HS + RTX 3060 con Python 3.11.9, PyTorch 2.3.1+cu121 y CUDA 12.1. Cada arquitectura procesó una señal determinística mono float32 de 60 s a 44.1 kHz. Se utilizaron chunks de 20,000 muestras, cinco iteraciones de warm-up y veinte repeticiones medidas. En CPU se fijó un thread de cómputo.

Para LSTM se preservó el hidden state entre bloques. CNN y TCN recibieron el contexto causal necesario para cada bloque. El tiempo reportado corresponde al procesamiento completo de la señal y se sincronizó la GPU antes y después de cada medición.

Las métricas de eficiencia son número de parámetros, tamaño estandarizado del state_dict, tiempo mediano, percentil 95 del tiempo, Real-Time Factor (RTF), múltiplo de tiempo real y memoria CUDA pico. Un RTF menor que 1 indica que el modelo procesa la señal más rápido que su duración real. El benchmark mide throughput y no debe interpretarse como latencia end-to-end de un sistema interactivo con interfaz, drivers y buffers.

### 6.1.3 Evaluación cualitativa y generalización

Esta parte del protocolo continúa pendiente de cierre con la asesora. Se propone realizar una prueba adicional con material claramente no visto durante entrenamiento y complementar las métricas objetivas mediante una evaluación perceptual online. Todavía no se han fijado el número de participantes, la escala de respuesta, los fragmentos de audio, el orden de presentación, la aleatorización ni el método de análisis. Por esta razón, este borrador no presenta resultados cualitativos como si ya hubieran sido ejecutados.

## 6.2 Validación de resultados

### 6.2.1 Estado de la matriz experimental

La matriz principal quedó completa con 24 modelos scratch, ocho CNN, ocho LSTM y ocho TCN. Al incluir los cuatro experimentos complementarios de fine-tuning, el manifest contiene 28 runs. El evaluador común procesó los 28 sin errores ni faltantes.

Los resultados completos por setting de la matriz principal se presentan en el Apéndice A. En esta sección se utilizan promedios entre configuraciones como resumen descriptivo, pero las conclusiones también consideran el comportamiento individual de cada setting.

### 6.2.2 Fidelidad en Starcaster Distortion

| Arquitectura | ESR | MAE | MR-STFT | Error LUFS (dB) |
|---|---:|---:|---:|---:|
| CNN | 0.976334 | **0.044169** | 1.921682 | **0.390750** |
| LSTM | 1.007093 | 0.046173 | 3.252119 | 0.852250 |
| TCN | **0.935714** | 0.045729 | **1.577064** | 0.701250 |

En Starcaster, la TCN obtiene el menor ESR y MR-STFT en los cuatro settings. La CNN obtiene el menor MAE y error LUFS en los cuatro. Esto significa que la TCN presenta menor error normalizado de energía y menor error tiempo-frecuencia, mientras la CNN mantiene una correspondencia más cercana en error absoluto muestra a muestra y loudness global.

La diferencia entre métricas es importante porque evita afirmar que la TCN sea superior en todas las dimensiones. El resultado depende de qué propiedad de la señal se utilice para definir fidelidad.

### 6.2.3 Fidelidad en MXR Dyna Comp

| Arquitectura | ESR | MAE | MR-STFT | Error LUFS (dB) |
|---|---:|---:|---:|---:|
| CNN | 0.967692 | **0.029390** | 2.301825 | **0.724500** |
| LSTM | **0.964000** | 0.030220 | **1.959813** | 5.215500 |
| TCN | 1.217671 | 0.035399 | 2.294639 | 4.285000 |

En Dyna Comp la comparación cambia. La LSTM obtiene el menor ESR promedio, aunque la diferencia frente a CNN es pequeña, y también obtiene el menor MR-STFT promedio. La CNN obtiene el menor MAE y una ventaja amplia en error LUFS. La TCN no presenta la misma ventaja de fidelidad observada en Starcaster.

El contraste entre MR-STFT y LUFS es especialmente relevante para el compresor. La LSTM puede presentar menor error espectral y, al mismo tiempo, una diferencia de loudness considerablemente mayor. Este resultado refuerza la decisión de mantener varias métricas separadas.

### 6.2.4 Comportamiento de validación en Dyna Comp

Los experimentos LSTM y TCN sobre Dyna Comp presentan un patrón que no aparece de la misma forma en Starcaster. En los cuatro settings de TCN el mejor checkpoint de validación aparece en epoch 1. En LSTM, los mejores checkpoints de Dyna Comp aparecen entre epochs 1 y 2. Después de esos puntos la loss de entrenamiento continúa disminuyendo, mientras la validación empeora o no vuelve a recuperar el mínimo inicial.

El patrón es consistente entre configuraciones y entre dos familias de modelos temporales. No se atribuye todavía a una causa única. Entre las explicaciones posibles se encuentran diferencias de distribución entre entrenamiento y validación, sensibilidad del compresor al nivel y estructura temporal del material, o una interacción entre la función de pérdida y la dinámica del efecto. Estas posibilidades deben presentarse como hipótesis y no como una causa demostrada.

### 6.2.5 Eficiencia computacional

| Arq. | Disp. | Params | State MB | Mediana (s) | RTF | xRT | CUDA MB |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN | CPU | 8,305 | 0.0317 | 6.463548 | 0.107726 | 9.28 | N/A |
| LSTM | CPU | 4,513 | 0.0172 | 2.359616 | 0.039327 | 25.43 | N/A |
| TCN | CPU | 43,873 | 0.1684 | 6.825922 | 0.113765 | 8.79 | N/A |
| CNN | GPU | 8,305 | 0.0317 | 0.912496 | 0.015208 | 65.75 | 42.40 |
| LSTM | GPU | 4,513 | 0.0172 | 1.110357 | 0.018506 | 54.04 | 80.68 |
| TCN | GPU | 43,873 | 0.1684 | 0.184466 | 0.003074 | 325.26 | 38.88 |

Las tres arquitecturas obtienen RTF menor que 1 en CPU y GPU bajo el protocolo utilizado. En CPU, LSTM presenta el mejor throughput y además es la arquitectura más pequeña. En GPU, TCN procesa 60 s de audio en una mediana de 0.184 s, aproximadamente 325 veces tiempo real. Este resultado muestra que el número de parámetros no puede utilizarse como sustituto directo del tiempo de inferencia.

![RTF mediano y número de parámetros por arquitectura](docs/figures/fidelity_efficiency/architecture_rtf_cpu_gpu.png)

### 6.2.6 Compromiso entre fidelidad y eficiencia

Para estudiar el compromiso entre fidelidad y costo se cruzó el RTF de cada arquitectura con los promedios de fidelidad entre 25 %, 50 %, 75 % y 100 %. Estos settings son condiciones experimentales distintas y no réplicas estadísticas. Por esta razón, las barras verticales muestran el rango mínimo a máximo observado y no una desviación estándar inferencial.

En Starcaster, la TCN se ubica simultáneamente más abajo y más a la izquierda en los planos ESR frente a RTF GPU y MR-STFT frente a RTF GPU. En estas dos dimensiones concretas presenta menor error y menor costo de inferencia que CNN y LSTM. La dominancia no se mantiene para MAE o LUFS, donde CNN conserva menor error.

![Starcaster: fidelidad frente a costo GPU](docs/figures/fidelity_efficiency/starcaster_esr_vs_cuda_rtf.png)

En Dyna Comp no aparece una arquitectura que domine de la misma manera. TCN conserva el menor RTF GPU, pero obtiene un ESR promedio mayor. LSTM obtiene menor ESR y MR-STFT promedio, mientras CNN presenta menor MAE y una diferencia de loudness mucho menor. La arquitectura más conveniente depende de qué dimensión de fidelidad se priorice.

![Dyna Comp: fidelidad frente a costo GPU](docs/figures/fidelity_efficiency/dynacomp_esr_vs_cuda_rtf.png)

### 6.2.7 Fuentes de error e incertidumbre

La interpretación de los resultados debe considerar varias fuentes de incertidumbre experimental. La primera es la cadena de adquisición. No se utilizó una reamp box, por lo que la interfaz entrega señal de línea directamente a los pedales. Esta condición se mantiene constante, pero su efecto eléctrico no se cuantificó de forma independiente.

La segunda es la alineación temporal. El pipeline mantiene intencionalmente un margen causal residual de 128 muestras, equivalente a 2.90 ms. Este valor es conocido y reproducible, pero no debe confundirse con una medición física de latencia del dispositivo.

La tercera corresponde al procesamiento de los targets del Dyna Comp. Se aplicó un Clip Gain fijo de +6 dB a toda la serie. El ajuste no cambia entre arquitecturas, por lo que no introduce una ventaja relativa para un modelo específico, pero forma parte de la definición del target utilizado en el experimento.

La adquisición se almacenó en PCM16 a 44.1 kHz. No se realizó una medición independiente del ruido de fondo ni del error de cuantización de la cadena, por lo que estos factores no se separan del error total reportado por las métricas.

Finalmente, el benchmark computacional presenta variación entre repeticiones. Para cuantificarla se registró el percentil 95 del tiempo. En CPU, los p95 fueron 6.788 s para CNN, 2.551 s para LSTM y 7.373 s para TCN. En GPU, fueron 1.309 s, 1.135 s y 0.223 s respectivamente. Por esta razón, las conclusiones de eficiencia se basan principalmente en medianas y se interpretan dentro del hardware y protocolo medidos.

# 7. CONCLUSIONES

## 7.1 Discusión

La matriz experimental principal permite comparar CNN, LSTM y TCN bajo los mismos efectos, configuraciones y divisiones de datos. El resultado más importante es que la arquitectura no puede evaluarse de forma aislada del efecto ni de la métrica utilizada para definir fidelidad.

Para Starcaster, TCN obtiene menor ESR y MR-STFT en las cuatro configuraciones. En GPU también presenta el menor RTF. Por lo tanto, cuando la comparación se plantea específicamente como ESR o MR-STFT frente a costo GPU, TCN ofrece el compromiso más favorable entre las tres arquitecturas evaluadas. Sin embargo, CNN conserva menor MAE y error LUFS, por lo que la ventaja de TCN no puede generalizarse a todas las dimensiones de fidelidad.

Para Dyna Comp la situación es diferente. LSTM obtiene el menor ESR y MR-STFT promedio, CNN obtiene menor MAE y una correspondencia de loudness considerablemente mejor, y TCN presenta el mejor throughput GPU a costa de un ESR y MAE mayores. Este resultado muestra un trade-off real y evita identificar un único ganador para el compresor.

El benchmark también muestra que parámetros y velocidad no son equivalentes. TCN tiene 43,873 parámetros, frente a 8,305 de CNN y 4,513 de LSTM, pero es la arquitectura más rápida en GPU. En CPU ocurre lo contrario y LSTM obtiene el mejor throughput. La plataforma de ejecución debe formar parte de la conclusión sobre eficiencia.

Las principales limitaciones son el uso de un único dispositivo físico por tipo de efecto, la ausencia de reamp box, el uso de cuatro posiciones del control principal y la falta todavía de evaluación perceptual y de una prueba adicional de generalización. Además, el comportamiento de validación temprana en Dyna Comp con LSTM y TCN está documentado, pero su causa no fue aislada experimentalmente.

Con la evidencia disponible, la respuesta preliminar a la pregunta de investigación es que no existe una arquitectura universalmente superior. Para Starcaster, TCN ofrece el mejor compromiso observado entre ESR/MR-STFT y throughput GPU. Para Dyna Comp, la selección depende de si se prioriza fidelidad espectral y ESR, correspondencia de amplitud y loudness, o velocidad de inferencia. En CPU, LSTM es la arquitectura más eficiente bajo el protocolo medido.

La validación cualitativa todavía no puede discutirse como un resultado del proyecto porque no se ha ejecutado. La conclusión final deberá incorporar esa evidencia junto con la prueba adicional sobre material no visto.

## 7.2 Trabajo futuro

Antes de cerrar el proyecto se deben completar dos actividades que hacen parte de la validación prevista. La primera es definir y ejecutar una prueba adicional sobre material claramente no visto para estudiar generalización de manera más explícita. La segunda es diseñar y ejecutar la evaluación perceptual, definiendo participantes, consentimiento, fragmentos, escala de respuesta, aleatorización y método de análisis. También queda por decidir con la asesora si se incorpora una métrica adicional de envolvente o dinámica para complementar la evaluación del compresor.

Como extensiones posteriores al alcance de esta tesis, sería útil estudiar modelos condicionados por los parámetros del efecto. En este trabajo se entrena una red diferente para cada setting, pero técnicas como FiLM permitirían investigar si una sola red puede representar múltiples configuraciones del pedal. Esta comparación introduciría una pregunta adicional sobre especialización frente a condicionamiento y por eso no se incluyó en la matriz principal actual.

También sería posible ampliar el estudio a más dispositivos físicos, incluir más controles por pedal, evaluar otras familias de modelos, medir latencia end-to-end dentro de un plugin de audio y repetir el benchmark sobre hardware adicional. Estas extensiones permitirían estudiar qué tan generales son los resultados obtenidos aquí.

# REFERENCIAS

Damskägg, E.-P., Juvela, L., Thuillier, E., & Välimäki, V. (2019). Deep learning for tube amplifier emulation. *Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 471–475.

Damskägg, E.-P., Juvela, L., & Välimäki, V. (2019). Real-time modeling of audio distortion circuits with deep learning. *Proceedings of the 16th Sound & Music Computing Conference*, 332–339.

Hawley, S. H., Colburn, B., & Mimilakis, S. I. (2019). SignalTrain: Profiling audio compressors with deep neural networks. *arXiv:1905.11928*.

Martínez Ramírez, M. A., Benetos, E., & Reiss, J. D. (2020). Deep learning for black-box modeling of audio effects. *Applied Sciences, 10*(2), 638. doi: 10.3390/app10020638.

Steinmetz, C. J., & Reiss, J. D. (2022). Efficient neural networks for real-time modeling of analog dynamic range compression. *152nd Convention of the Audio Engineering Society*.

Wright, A., Damskägg, E.-P., & Välimäki, V. (2019). Real-time black-box modelling with recurrent neural networks. *Proceedings of the 22nd International Conference on Digital Audio Effects (DAFx-19)*, 1–8.

Wright, A., Damskägg, E.-P., Juvela, L., & Välimäki, V. (2020). Real-time guitar amplifier emulation with deep learning. *Applied Sciences, 10*(3), 766. doi: 10.3390/app10030766.

Wright, A., & Välimäki, V. (2020). Perceptual loss function for neural modeling of audio systems. *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 251–255. doi: 10.1109/ICASSP40776.2020.9052944.

# APÉNDICE A. RESULTADOS COMPLETOS POR CONFIGURACIÓN

Las siguientes tablas presentan los resultados de los 24 modelos principales evaluados con el protocolo común. Los valores corresponden al checkpoint seleccionado por validación y al split de prueba. Todas las métricas de error se interpretan como menor es mejor.

## A.1 Starcaster Distortion

| Arquitectura | Setting | ESR | MAE | MR-STFT | Error LUFS (dB) |
|---|---:|---:|---:|---:|---:|
| CNN | 25 | 0.662231 | 0.030073 | 1.760229 | 0.343 |
| CNN | 50 | 1.008393 | 0.043079 | 1.597457 | 0.370 |
| CNN | 75 | 1.096581 | 0.051456 | 2.279925 | 0.670 |
| CNN | 100 | 1.138131 | 0.052067 | 2.049116 | 0.180 |
| LSTM | 25 | 0.673416 | 0.031690 | 3.082182 | 0.905 |
| LSTM | 50 | 1.035320 | 0.044916 | 2.841576 | 0.686 |
| LSTM | 75 | 1.179052 | 0.054895 | 4.051253 | 1.349 |
| LSTM | 100 | 1.140583 | 0.053191 | 3.033465 | 0.469 |
| TCN | 25 | 0.611206 | 0.031685 | 1.305833 | 0.695 |
| TCN | 50 | 0.949926 | 0.044016 | 1.381391 | 0.842 |
| TCN | 75 | 1.057159 | 0.053097 | 1.994311 | 0.867 |
| TCN | 100 | 1.124564 | 0.054118 | 1.626723 | 0.401 |

## A.2 MXR Dyna Comp

| Arquitectura | Setting | ESR | MAE | MR-STFT | Error LUFS (dB) |
|---|---:|---:|---:|---:|---:|
| CNN | 25 | 0.997619 | 0.027591 | 1.529316 | 0.184 |
| CNN | 50 | 0.854788 | 0.027545 | 2.424215 | 1.592 |
| CNN | 75 | 1.067203 | 0.031741 | 2.353690 | 0.262 |
| CNN | 100 | 0.951159 | 0.030684 | 2.900078 | 0.860 |
| LSTM | 25 | 1.127779 | 0.031221 | 1.818048 | 6.724 |
| LSTM | 50 | 0.903959 | 0.029044 | 1.902550 | 4.038 |
| LSTM | 75 | 0.959539 | 0.031079 | 2.027137 | 5.373 |
| LSTM | 100 | 0.864724 | 0.029537 | 2.091518 | 4.727 |
| TCN | 25 | 1.269825 | 0.034842 | 2.192171 | 4.048 |
| TCN | 50 | 1.209783 | 0.035498 | 2.273650 | 4.244 |
| TCN | 75 | 1.205342 | 0.035554 | 2.321545 | 4.417 |
| TCN | 100 | 1.185735 | 0.035702 | 2.391191 | 4.431 |
