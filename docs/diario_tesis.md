**# Diario técnico y log maestro — Proyecto de grado**



**\*\*Proyecto:\*\*** Modelado de efectos de compresión y distorsión de guitarra eléctrica mediante técnicas de aprendizaje profundo  

**\*\*Estudiante:\*\*** Ponto Andrés Moreno Trejos  

**\*\*Asesora:\*\*** Haydemar Maria Nuñez Castro  

**\*\*Programa / área:\*\*** Ingeniería de Sistemas y Computación — Inteligencia Artificial y Procesamiento Digital de Señales  

**\*\*Repositorio principal:\*\*** \`https\://github.com/pontoandres/Tesis-Audio-DL.git\`  

**\*\*Última actualización de este log:\*\*** 2026-10-13  

**\*\*Estado del documento:\*\*** vivo / acumulativo. No borrar entradas históricas; corregirlas mediante notas posteriores cuando haga falta.



\---



**## 0. Propósito de este archivo**



Este archivo es la **\*\*fuente de verdad narrativa del proyecto\*\***. Su objetivo es registrar, con el mayor nivel de detalle razonable:



\- decisiones de alcance;

\- cambios metodológicos;

\- reuniones y acuerdos con la asesora;

\- adquisición de datos;

\- configuración de hardware y software;

\- estructura del repositorio;

\- scripts desarrollados;

\- errores encontrados;

\- hipótesis descartadas;

\- decisiones de diseño;

\- comandos relevantes;

\- experimentos;

\- métricas;

\- resultados;

\- problemas de reproducibilidad;

\- próximos pasos;

\- material reutilizable para el documento final.



La idea es que el informe de tesis no tenga que reconstruirse de memoria al final. Las secciones de metodología, implementación, resultados, discusión y limitaciones se podrán redactar tomando este log como insumo.



**### Convención de etiquetas**



\- \`[ALCANCE]\`: decisiones sobre qué entra o no entra en el proyecto.

\- \`[METODOLOGÍA]\`: procedimiento experimental o criterio reproducible.

\- \`[DATASET]\`: adquisición, limpieza, organización y propiedades de datos.

\- \`[IMPLEMENTACIÓN]\`: código, arquitectura, librerías, repositorios y configuración.

\- \`[EXPERIMENTO]\`: entrenamiento o evaluación concreta.

\- \`[RESULTADO]\`: métrica o hallazgo medido.

\- \`[PROBLEMA]\`: error o comportamiento inesperado.

\- \`[FIX]\`: corrección implementada.

\- \`[DECISIÓN]\`: elección que debe conservarse para reproducibilidad.

\- \`[GIT]\`: repositorios, ramas, submódulos, commits o estructura.

\- \`[ASESORA]\`: acuerdos, comentarios o entregables relacionados con reuniones.

\- \`[PENDIENTE]\`: trabajo aún no realizado.

\- \`[DOCUMENTO]\`: material que puede convertirse casi directamente en texto de tesis.



\---



**# 1. Definición vigente del proyecto**



**## 1.1 Título**



**\*\*Modelado de efectos de compresión y distorsión de guitarra eléctrica mediante técnicas de aprendizaje profundo.\*\***



\> Nota crítica: **\*\*el compresor sigue dentro del alcance del proyecto\*\***. No se debe eliminar del plan. Los dos efectos físicos considerados son distorsión y compresión.



**## 1.2 Pregunta de investigación**



¿Cuál de las arquitecturas de aprendizaje profundo evaluadas ofrece el mejor equilibrio entre fidelidad de la señal y complejidad computacional para modelar efectos de compresión y distorsión de guitarra eléctrica a partir de ejemplos de entrada y salida?



**## 1.3 Arquitecturas**



Se compararán:



\- CNN;

\- TCN;

\- LSTM.



La intención no es diseñar tres arquitecturas completamente desde cero, sino **\*\*seleccionar, adaptar y entrenar implementaciones representativas y reproducibles\*\*** utilizadas en modelado neuronal de audio.



**## 1.4 Efectos y controles**



**### Distorsión**

Pedal: **\*\*Fender Starcaster Distortion\*\***.



Control principal a variar:

\- \`Distortion\` / ganancia.



Controles secundarios:

\- mantener fijos nivel y ecualización / tono durante una serie comparable.



Configuraciones previstas en propuesta:

\- 0%;

\- 25%;

\- 50%;

\- 75%;

\- 100%.



Estado real al 2026-10-13:

\- 25% grabado, procesado y CNN entrenada;

\- 50% grabado, procesado y CNN entrenada;

\- 75% grabado, procesado y CNN entrenada;

\- 100% grabado, procesado y CNN entrenada;

\- 0%: estado experimental aún por confirmar antes de considerarlo completado.



**### Compresión**

Pedal: **\*\*MXR Dyna Comp\*\***.



Control principal a variar:

\- \`Sensitivity\`.



Control secundario:

\- nivel de salida fijo.



Estado al 2026-10-13:

\- las grabaciones del compresor **\*\*todavía no se han realizado\*\***;

\- continúa siendo parte obligatoria del alcance.



**## 1.5 Métricas previstas**



**### Fidelidad**

\- ESR (Error-to-Signal Ratio);

\- MAE o MSE;

\- diferencias espectrales mediante STFT;

\- apoyo cualitativo con forma de onda, espectro y escucha de ejemplos.



**### Eficiencia**

\- número de parámetros;

\- tamaño del modelo;

\- tiempo de inferencia;

\- factor de tiempo real (RTF);

\- uso aproximado de recursos durante inferencia.



**### Criterio de comparabilidad**

Las mediciones finales de eficiencia deben hacerse en un **\*\*mismo entorno de hardware y software\*\***, preferiblemente en la laptop RTX 3060, para no mezclar resultados de máquinas diferentes.



\---



**# 2. Hardware y software**



**## 2.1 Hardware de adquisición**



\- Guitarra principal: **\*\*Ibanez RG350DX\*\***.

\- Interfaz: **\*\*Native Instruments Komplete Audio 6\*\***.

\- Pedal de distorsión: **\*\*Fender Starcaster Distortion\*\***.

\- Pedal de compresión: **\*\*MXR Dyna Comp\*\***.

\- No se dispone de reamp box / thru dedicado.



**## 2.2 Computadores**



**### Laptop principal**

ASUS ROG Zephyrus G14 GA401QM.



\- CPU: AMD Ryzen 7 5800HS.

\- RAM: 16 GB.

\- GPU: NVIDIA GeForce RTX 3060.

\- Sistema: Windows.

\- Entorno usado para entrenamiento principal.



Python detectados históricamente:

\- 3.12;

\- 3.11;

\- 3.9.



Entorno del proyecto:

\- \`.venv\`;

\- Python 3.11.9.



PyTorch:

\- \`torch 2.3.1+cu121\`;

\- CUDA comprobado y funcional.



Ruta del proyecto:

\`C:\Users\ponto\Documents\Tesis-Audio-DL\`



**### Desktop antiguo**

\- CPU: Intel i5-6500.

\- GPU: GTX 1060 3 GB.

\- Python 3.13 instalado en sistema.

\- venv usado con Python 3.11.9 para compatibilidad.



Ruta del proyecto / trabajos tempranos:

\`C:\Users\pontomoreno\Documents\Tesis-Audio-DL\`



Se decidió preferir la laptop RTX 3060 para entrenamiento. El desktop puede servir para grabación, inspección, inferencia o pruebas pequeñas.



**## 2.3 Software de audio**



\- Ableton Live Lite.

\- Audacity usado / considerado para inspección y conversiones.

\- Driver principal de captura: Komplete Audio ASIO Driver.

\- Frecuencia de muestreo final de trabajo: **\*\*44.1 kHz\*\***.

\- Buffer usado en sesión validada: **\*\*512 samples\*\***.

\- Driver Error Compensation: **\*\*0 ms\*\***.



\---



**# 3. Protocolo de adquisición de audio**



**## 3.1 Cadena física usada para Starcaster**



Cadena implementada:



\`Komplete Audio OUT 1 -> pedal -> Komplete Audio IN 3\`



En Ableton:



**### Track de reproducción dry**

\- Audio From: \`No Input\`;

\- Audio To: \`Ext. Out 1\`;

\- contiene el WAV dry original;

\- Warp: **\*\*OFF\*\***;

\- Clip Gain: \`0 dB\`;

\- sin plugins.



**### Track wet**

\- Audio From: \`Ext. In 3\`;

\- Monitor: \`Off\`;

\- Audio To: \`Main\`;

\- armado para grabar.



**### Condiciones que deben mantenerse**

\- mismo nivel de salida de la interfaz;

\- mismo nivel de entrada;

\- mismos controles secundarios del pedal;

\- solamente cambia el control principal del efecto;

\- no cambiar cadena a mitad de una familia de capturas.



**## 3.2 Razón para conservar el dry original**



El archivo dry original es la entrada del dataset. No se reemplaza por una regrabación del dry. La captura nueva solo produce el \`target\` procesado.



**## 3.3 Limitación de reamp**



La interfaz está enviando señal de línea hacia la entrada del pedal, en lugar de usar una reamp box dedicada. No es la condición eléctricamente ideal de un reamp profesional, pero:



1\. se mantuvo la cadena constante entre configuraciones;

2\. el objetivo principal es una comparación controlada dentro del proyecto;

3\. cambiar la cadena a mitad del dataset sería peor para comparabilidad.



Esta limitación debe aparecer en el documento final.



\---



**# 4. Repositorios y estructura del proyecto**



**## 4.1 Repositorio principal**



\`https\://github.com/pontoandres/Tesis-Audio-DL.git\`



Ruta laptop:

\`C:\Users\ponto\Documents\Tesis-Audio-DL\`



Estructura base creada:



\`\`\`text

Tesis-Audio-DL/

├── data/

│   ├── raw/

│   ├── processed/

│   └── starcaster/

├── scripts/

├── configs/

├── experiments/

│   └── cnn/

├── results/

└── external/

    └── Automated-GuitarAmpModelling/

\`\`\`



**## 4.2 Wright / Automated-GuitarAmpModelling**



Repositorio original:

\`https\://github.com/Alec-Wright/Automated-GuitarAmpModelling.git\`



Fork:

\`https\://github.com/pontoandres/Automated-GuitarAmpModelling.git\`



Submódulo correcto:

\`external/Automated-GuitarAmpModelling\`



Existió además un clone antiguo separado en el desktop:

\`C:\Users\pontomoreno\Documents\Tesis-Audio-DL\Automated-GuitarAmpModelling\`



No confundirlo con el submódulo usado por el proyecto.



Historial de referencias de commits registradas:

\- Wright original / submódulo temprano: \`e3146386...\`

\- CoreAudioML: \`bad9469...\`

\- commit del fork con compatibilidad / adaptación: \`3cd9184 Support external datasets and current PyTorch\`

\- main repo apuntando al fork modificado: \`f00b116 Use customized Wright model repository\`

\- commit temprano de estructura principal: \`694e998\`

\- commit temprano agregando submódulo Wright: \`ec5ad1d\`



**## 4.3 \`.gitignore\` relevante**



Se mantiene ignorado:



\`\`\`gitignore

/Automated-GuitarAmpModelling/

.venv/

\_\_pycache\_\_/

\*.pyc

data/

recordings/

\*.wav

\*.mp3

\*.mp4

\*.flac

\*.pt

\*.pth

\*.ckpt

\*.zip

\*.7z

\`\`\`



Debido a que \`data/\` y \`\*.wav\` están ignorados, las grabaciones que sí se deciden versionar deben agregarse con \`git add -f\`.



**## 4.4 WAVs grandes y GitHub**



Al subir los WAV de 75% y 100%, GitHub emitió advertencias por archivos mayores de 50 MB.



Ejemplos:

\- train 100: \~64.78 MB;

\- train 75: \~56.92 MB.



GitHub aceptó el push porque los archivos seguían por debajo del límite duro de archivos normales, pero se recomendó Git LFS.



Decisión actual:

\- no migrar todavía todo a LFS durante el avance urgente;

\- dejar documentada la advertencia;

\- evaluar estrategia final de almacenamiento del dataset antes de la entrega.



\---



**# 5. Cambios aplicados al repositorio de Wright**



Se realizaron adaptaciones necesarias para usar datasets externos y PyTorch actual.



**## 5.1 Scheduler**



Se removió el argumento no soportado:



\`\`\`python

verbose=True

\`\`\`



del scheduler correspondiente.



**## 5.2 Ruta del dataset**



Se cambió el uso fijo:



\`\`\`python

dataset.DataSet(data_dir='Data')

\`\`\`



por:



\`\`\`python

dataset.DataSet(data_dir=args.data_location)

\`\`\`



para entrenar con carpetas externas como:



\`\`\`text

..\\..\data\processed\starcaster\25

\`\`\`



**## 5.3 Conversión Tensor -> NumPy**



Se corrigió:



\`\`\`python

test_output.cpu().numpy()

\`\`\`



por:



\`\`\`python

test_output.detach().cpu().numpy()

\`\`\`



para evitar error al convertir tensores que requieren gradiente.



**## 5.4 OOM en evaluación final**



Problema:

\- después de terminar el entrenamiento de Starcaster 50, la evaluación final podía fallar por falta de memoria GPU;

\- el script retenía objetos / tensores al evaluar modelo final y luego best model.



Fix implementado:

\- \`network.eval()\`;

\- bloques \`torch.no_grad()\`;

\- guardar evaluación final;

\- \`del test_output\`;

\- \`del test_loss\`;

\- \`del test_loss_ESR\`;

\- \`del network\`;

\- \`torch.cuda.empty_cache()\`;

\- recargar \`model_best\`;

\- volver a \`eval()\` y \`no_grad()\`.



El arreglo quedó verificado posteriormente con Starcaster 25: el run completó evaluación de modelo final y best model sin OOM.



**## 5.5 Evaluador auxiliar**



Se creó / usó \`evaluate_model.py\` para evaluar Starcaster 50 después del OOM del flujo original.



No es necesario usarlo para runs futuros si \`dist_model.py\` termina correctamente y guarda las métricas de test.



\---



**# 6. Dataset Starcaster**



**## 6.1 Organización raw**



Ejemplo:



\`\`\`text

data/starcaster/train/

    starcaster_input_train.wav

    starcaster_25_target_44_train.wav

    starcaster_50_target_44_train.wav

    starcaster_75_target_44_train.wav

    starcaster_100_target_44_train.wav



data/starcaster/val/

    starcaster_input_val.wav

    starcaster_25_target_44_val.wav

    starcaster_50_target_44_val.wav

    starcaster_75_target_44_val.wav

    starcaster_100_target_44_val.wav



data/starcaster/test/

    starcaster_input_test.wav

    starcaster_25_target_44_test.wav

    starcaster_50_target_44_test.wav

    starcaster_75_target_44_test.wav

    starcaster_100_target_44_test.wav

\`\`\`



**## 6.2 Duraciones dry aproximadas**



\- train dry: 340 s;

\- val dry: 51.06 s;

\- test dry: 60 s.



El dataset no usa miles de archivos cortos: cada split se maneja principalmente como un WAV largo que posteriormente es segmentado por el loader / pipeline de entrenamiento.



\---



**# 7. Alineación temporal: evolución de la metodología**



**## 7.1 Problema identificado**



Los pares dry / wet no quedan sample-aligned automáticamente después del round trip de reproducción y captura.



El repositorio Wright consume pares directamente y no corrige latencia física por sí mismo.



Esto explicó en parte los pobres resultados de las primeras CNN del Starcaster.



**## 7.2 \`check_alignment.py\`**



Se desarrolló:



\`scripts/check_alignment.py\`



Funciones principales:

\- cargar WAV input y target;

\- verificar sample rate;

\- convertir a mono / float;

\- buscar una ventana activa;

\- remover DC;

\- correlación cruzada con SciPy;

\- detectar offset temporal;

\- permitir detectar inversión de polaridad usando \`abs(correlation)\`.



Convención:



\`\`\`text

lag > 0  -> TARGET está retrasado respecto a INPUT

lag < 0  -> TARGET está adelantado respecto a INPUT

\`\`\`



Corrección:

\- \`lag > 0\`: cortar comienzo de target;

\- \`lag < 0\`: cortar comienzo de input.



**## 7.3 Inversión de polaridad**



Las capturas wet presentan correlación negativa respecto al dry.



Decisión:

\- **\*\*NO invertir el target\*\***;

\- el modelo debe aprender la transformación física completa observada, incluida la inversión de polaridad.



**## 7.4 Error metodológico inicial corregido**



Se detectó que para crear un margen causal se estaba interpretando de forma incorrecta qué señal recortar.



Implementación final:



\`\`\`python

def add_causal_delay(input_audio, target_audio, delay):

    if delay > 0:

        input_audio = input_audio[delay:]



    length = min(len(input_audio), len(target_audio))

    return input_audio[:length], target_audio[:length]

\`\`\`



Interpretación:

\- se hace que el target quede aproximadamente \`delay\` muestras después del input;

\- esto proporciona a una red causal contexto previo del input para producir el target.



**## 7.5 Criterio causal**



Se escogió un margen objetivo aproximado de **\*\*128 muestras\*\***.



A 44.1 kHz:



\`\`\`text

128 / 44100 ≈ 2.90 ms

\`\`\`



Importante:

\- 128 no se presenta como “la latencia física verdadera”;

\- es un **\*\*criterio de alineación causal / margen de seguridad\*\*** usado de forma reproducible después de la alineación gruesa.



\---



**# 8. \`align_dataset.py\`**



Script:

\`scripts/align_dataset.py\`



Responsabilidades:



1\. cargar par raw;

2\. aplicar coarse alignment;

3\. aplicar causal delay adicional;

4\. convertir ambos archivos a PCM16;

5\. guardar estructura esperada por Wright.



Salida:



\`\`\`text

data/processed/starcaster/\<config>/\<split>/

    starcaster\<config>-input.wav

    starcaster\<config>-target.wav

\`\`\`



Los raw originales no se modifican.



**## 8.1 Conversión PCM**



Dry antiguos:

\- int16.



Wet antiguos 25/50:

\- int32 leído por SciPy, correspondiente a grabaciones de mayor profundidad de bits.



Wet nuevos 75/100:

\- int16.



El script convierte todo a PCM16 al final para homogeneizar lo que consume el modelo.



No se hace normalización independiente de amplitud por split porque la distorsión es no lineal y el nivel forma parte del mapeo físico.



\---



**# 9. Coarse alignment conocido**



**## 9.1 25% y 50%**



Valores históricos usados:



\`\`\`text

25 train  = -73142

50 train  = -73139



25 val    = +95

50 val    = +97



25 test   = +109

50 test   = -110595

\`\`\`



**## 9.2 75%**



Después de ampliar la búsqueda de \`check_alignment.py\` a ±5 s:



\`\`\`text

75 train = -73137 samples

          -1658.4354 ms

          corr -0.4736



75 val   = -97806 samples

          -2217.8231 ms

          corr -0.5169



75 test  = -110589 samples

          -2507.6871 ms

          corr -0.5322

\`\`\`



**## 9.3 100%**



Primera grabación de train 100%:

\- duración target: 385.071 s;

\- input: 340.000 s;

\- correlación mala: aproximadamente \`+0.0534\`;

\- resultado de alignment incoherente.



Diagnóstico:

\- **\*\*Warp estaba activado accidentalmente en Ableton\*\***.



Acción:

\- se descartó / eliminó el WAV defectuoso;

\- se repitió la captura con Warp OFF;

\- se resolvió conflicto Git por delete remoto vs versión nueva local.



Grabación corregida:



\`\`\`text

100 train = -73136 samples

            -1658.4127 ms

            corr -0.4738



100 val   = -97806 samples

            -2217.8231 ms

            corr -0.5142



100 test  = -110589 samples

            -2507.6871 ms

            corr -0.5318

\`\`\`



Hallazgo:

\- train 75 y train 100 difieren solo en 1 sample de coarse lag;

\- val 75/100 coinciden;

\- test 75/100 coinciden.



Esto respalda consistencia de captura.



\---



**# 10. \`check_drift.py\`**



Script:

\`scripts/check_drift.py\`



Método:

\- ventanas de 4 s;

\- búsqueda ±100 ms;

\- posiciones distribuidas:

  - 5%;

  - 25%;

  - 50%;

  - 75%;

  - 95%.



Debido a la inversión de polaridad, se usa:



\`\`\`python

j = int(np.argmin(c2))

\`\`\`



y **\*\*no\*\*** \`argmax\`.



Se parametrizó para poder elegir pedal y configuraciones, por ejemplo:



\`\`\`python

PEDAL = "starcaster"

CONFIGS = ("75", "100")

SPLITS = ("train", "val", "test")

\`\`\`



La intención es reutilizarlo posteriormente para el compresor.



\---



**# 11. Causal delay final**



**## 11.1 25%**



\`\`\`text

train = 144

val   = 134

test  = 132

\`\`\`



**## 11.2 50%**



\`\`\`text

train = 149

val   = 135

test  = 127

\`\`\`



**## 11.3 75% — proceso**



Con delay provisional 128 se midió:



\`\`\`text

TRAIN: 117, 124, 103, 103, 103

VAL:   130, 128, 119, 119, 121

TEST:  128, 133, 123, 110, 108

\`\`\`



Medianas:



\`\`\`text

TRAIN = 103

VAL   = 121

TEST  = 123

\`\`\`



Se aplicó:



\`\`\`text

nuevo_delay = delay_actual + (128 - mediana_residual)

\`\`\`



Resultado final:



\`\`\`text

75 train = 153

75 val   = 135

75 test  = 133

\`\`\`



Verificación final:



\`\`\`text

TRAIN: 142, 149, 128, 128, 128 -> mediana 128

VAL:   137, 135, 126, 126, 128 -> mediana 128

TEST:  133, 138, 128, 115, 113 -> mediana 128

\`\`\`



**## 11.4 100% — proceso**



Con delay provisional 128:



\`\`\`text

TRAIN: 117, 124, 102, 103, 102

VAL:   130, 128, 119, 119, 121

TEST:  128, 134, 124, 110, 108

\`\`\`



Resultado final:



\`\`\`text

100 train = 153

100 val   = 135

100 test  = 132

\`\`\`



Verificación final:



\`\`\`text

TRAIN: 142, 149, 127, 128, 127 -> mediana 128

VAL:   137, 135, 126, 126, 128 -> mediana 128

TEST:  132, 138, 128, 114, 112 -> mediana 128

\`\`\`



**## 11.5 Interpretación del drift**



Hay variaciones de aproximadamente 20 samples entre ventanas, equivalentes a \~0.45 ms.



Decisión:

\- aceptar esta variación;

\- usar mediana como criterio robusto;

\- congelar preprocessing de 75/100;

\- no seguir “persiguiendo” cada ventana individual.



\---



**# 12. Duraciones procesadas finales conocidas**



**## 12.1 25%**



\`\`\`text

train: 14,817,456 samples = 335.997 s

val:    2,251,524 samples =  51.055 s

test:   2,645,759 samples =  59.995 s

\`\`\`



**## 12.2 50%**



\`\`\`text

train: 14,920,604 samples = 338.336 s

val:    2,251,618 samples =  51.057 s

test:   2,535,279 samples =  57.489 s

\`\`\`



**## 12.3 75%**



\`\`\`text

train: 14,920,599 samples = 338.336 s

val:    2,153,716 samples =  48.837 s

test:   2,535,164 samples =  57.487 s

\`\`\`



**## 12.4 100%**



\`\`\`text

train: 14,920,599 samples = 338.336 s

val:    2,153,716 samples =  48.837 s

test:   2,535,165 samples =  57.487 s

\`\`\`



Observación:

\- 75 y 100 son prácticamente idénticos en longitud después del preprocessing;

\- test difiere solo en 1 sample por el causal delay elegido.



\---



**# 13. \`compare_splits.py\`**



Objetivo:

\- verificar estadísticamente train / validation / test antes del entrenamiento;

\- no es una métrica final del modelo.



Calcula:

\- duración;

\- número de samples;

\- RMS input;

\- RMS target;

\- peak;

\- diferencia RMS target/input;

\- correlación directa;

\- scalar lineal óptimo;

\- ESR de un baseline lineal.



Advertencia metodológica:

\- debido al margen causal deliberado (\~128 samples), la correlación directa same-index, el scalar y el ESR lineal no deben interpretarse como medidas fuertes de fidelidad física;

\- RMS y peak son especialmente útiles para detectar diferencias de nivel entre splits.



**## 13.1 Starcaster 25**



\`\`\`text

TRAIN

Duration: 335.997 s

Input RMS:  -23.10 dBFS

Input peak:  -5.18 dBFS

Target RMS: -24.26 dBFS

Target peak:-17.02 dBFS

Target/Input RMS difference: -1.16 dB

Correlation: +0.1032

Optimal scalar: +0.0903

Linear ESR: 0.9893



VAL

Duration: 51.055 s

Input RMS:  -23.50 dBFS

Input peak:  -5.16 dBFS

Target RMS: -25.13 dBFS

Target peak:-17.89 dBFS

Target/Input RMS difference: -1.63 dB

Correlation: -0.0688

Optimal scalar: -0.057

Linear ESR: 0.9953



TEST

Duration: 59.995 s

Input RMS:  -19.75 dBFS

Input peak:  -4.95 dBFS

Target RMS: -23.84 dBFS

Target peak:-17.19 dBFS

Target/Input RMS difference: -4.08 dB

Correlation: +0.2163

Optimal scalar: +0.1352

Linear ESR: 0.9532

\`\`\`



**## 13.2 Starcaster 50**



Datos relevantes registrados:



\`\`\`text

TRAIN

Input RMS:  -23.13 dBFS

Target RMS: -23.70 dBFS

Target/Input: -0.57 dB



VAL

Input RMS:  -23.50 dBFS

Target RMS: -24.45 dBFS

Target/Input: -0.95 dB



TEST

Input RMS:  -19.57 dBFS

Target RMS: -23.09 dBFS

Target/Input: -3.53 dB

\`\`\`



**## 13.3 Hallazgo de distribución**



El dry de test es aproximadamente 3.3–4 dB más fuerte que train / val.



Este patrón aparece tanto en 25 como en 50, por lo que no parece un error aislado de una configuración del pedal.



Implicación:

\- un efecto no lineal depende del nivel de entrada;

\- test puede estar funcionando parcialmente como una prueba de cambio de distribución en amplitud;

\- esto puede contribuir a que la generalización a test sea peor que validation.



Decisión actual:

\- **\*\*no normalizar cada split de forma independiente\*\***, porque destruiría parte de la relación física input/output;

\- mantener consistencia del dry entre configuraciones del Starcaster;

\- discutir posteriormente si el test se presenta como evaluación principal o como condición con amplitude shift / OOD.



**## 13.4 Starcaster 75 y 100**



Antes de ejecutar el sanity check final se detectó una inconsistencia de formato en las nuevas capturas 75/100.



[PROBLEMA] `compare_splits.py` falló porque el input procesado era mono, con forma `(N,)`, mientras que algunos targets nuevos habían sido exportados desde Ableton como stereo, con forma `(N, 2)`.



El error observado fue equivalente a:



```text

ValueError: all the input array dimensions except for the concatenation axis must match exactly

input size ≈ 14.9 M samples

target second dimension = 2

```



El problema no correspondía a una pérdida o corrupción de la grabación. `check_drift.py` ya había utilizado explícitamente el canal 0 de estas señales y había producido resultados coherentes.



[DECISIÓN] No modificar `align_dataset.py` para corregir este caso, dado que el stereo fue un error de exportación/adquisición y no una característica deseada del pipeline.



[FIX] Los targets 75/100 fueron convertidos offline a mono conservando el canal 0. Posteriormente se regeneró el dataset procesado.



Después de la conversión, `align_dataset.py` y `compare_splits.py` completaron correctamente.



Resultados relevantes del sanity check:



```text

75%

train input RMS ≈ -23.13 dBFS

val input RMS   ≈ -23.41 dBFS

test input RMS  ≈ -19.57 dBFS

test target RMS ≈ -22.53 dBFS

test target/input ≈ -2.96 dB



100%

train input RMS ≈ -23.13 dBFS

val input RMS   ≈ -23.41 dBFS

test input RMS  ≈ -19.57 dBFS

test target RMS ≈ -22.49 dBFS

test target/input ≈ -2.92 dB

```



[RESULTADO] Se vuelve a observar que el dry de test entra aproximadamente 3.5 dB más fuerte que train/validation.



[RESULTADO] No se detectó clipping relevante ni otra anomalía que justificara repetir las grabaciones.



[DECISIÓN] Los datasets procesados de 75% y 100% se consideran válidos para entrenamiento.



[PENDIENTE] Confirmar que las versiones mono de los WAV raw 75/100 hayan quedado reemplazadas también en GitHub.



\---



**# 14. Arquitectura CNN usada**



Implementación:

\`GatedConvNet\` del repositorio Wright.



Configuración usada en los runs actuales:



\`\`\`text

layers per block = 9

blocks = 2

channels / feature maps = 8

kernel_size = 3

dilation_growth = 2

batch_size = 8

epochs max = 100

validation_f = 1

cuda = 1

\`\`\`



Se ha descrito el frame / segment length como:

\- 22050 samples;

\- 0.5 s a 44.1 kHz.



Learning rate inicial observado / usado en configuración del repo:

\- 0.005.



Antes de escribir cifras arquitectónicas definitivas en el informe, verificar directamente la configuración/código del run final.



La entrada y salida son señales waveform mono en dominio temporal:

\- no se usan tokens;

\- la red predice muestras continuas de amplitud.



Estimaciones preliminares anotadas:

\- 8305 parámetros para esta configuración, verificados a partir de los checkpoints finales;

\- receptive field aproximado \~2045 samples (\~46.4 ms), sujeto a verificación exacta de implementación.



\---



**# 15. Control con dataset de Wright / Big Muff**



Se entrenó el mismo tipo de GatedConvNet usando el dataset de demostración de Wright como control.



Resultado aproximado registrado:



\`\`\`text

train loss: \~0.95575 -> \~0.1053

val loss:   \~0.727619 -> \~0.101598

best val:   \~0.101195

best epoch: \~96

\`\`\`



Interpretación:

\- la arquitectura y el código son capaces de aprender una distorsión conocida bajo un dataset bien preparado;

\- esto fue evidencia importante para enfocar el problema del Starcaster en datos/alineación, no simplemente en “la CNN no sirve”.



\---



**# 16. Experimentos Starcaster 50**



**## 16.1 Primer CNN corto — antes de la solución completa de alineación**



Run:

\`starcaster50_cnn20\`



20 epochs.



Resultados:

\- train loss aproximadamente \`0.9938 -> 0.7101\`;

\- best validation loss \`0.7039083242416382\`;

\- best epoch: 6;

\- \`test_loss: 0\` porque la evaluación final no se completó correctamente.



Inspección:

\- \`best_val_out.wav\` era casi silencioso;

\- output CNN no reproducía bien la señal del pedal.



Niveles registrados en ese momento:

\- dry RMS \~ -23.5 dBFS;

\- pedal RMS \~ -24.4 dBFS;

\- CNN output RMS \~ -57.4 dBFS.



Conclusión:

\- el problema no era simplemente nivel global dry/target;

\- se priorizó investigar alineación temporal.



**## 16.2 CNN causal inicial**



Run:

\`starcaster50_cnn100_causal128_v1-\`



Características:

\- coarse alignment;

\- causal margin \~128;

\- GatedConvNet;

\- hasta 100 epochs.



Comportamiento:

\- se detuvo alrededor de epoch 27;

\- train mejoró aproximadamente de \`0.832\` a \`0.165\`;

\- validation fue mejor al comienzo (\~0.593 epoch 1) y luego empeoró hacia \~0.84.



Esto mostró que el coarse alignment ayudaba entrenamiento pero aún faltaba ajuste fino.



**## 16.3 CNN con fine alignment**



Run:

\`starcaster50_cnn100_causal128_finealign_v1-\`



Comando base:



\`\`\`cmd

python dist_model.py --load_config "" --data_location "..\\..\data\processed\starcaster\50" --file_name "starcaster50" --device starcaster50_cnn100_causal128_finealign_v1 --model GatedConvNet --epochs 100 --validation_f 1 --batch_size 8 --cuda 1 --save_location "..\\..\results"

\`\`\`



Resultados:

\- \`current_epoch\`: \~87;

\- early stopping;

\- train: \~1.2058 -> \~0.1303;

\- best val: \~0.26149;

\- best epoch: \~61;

\- validation posterior \~0.26–0.27.



La evaluación final del script original sufrió OOM, por lo que se usó \`evaluate_model.py\`.



Test independiente del \`model_best\`:



\`\`\`text

test_loss = 0.9315442442893982

test_ESR  = 1.008393406867981

\`\`\`



Interpretación:

\- fuerte mejora de validation frente al run inicial;

\- test continúa siendo muy malo;

\- la diferencia de nivel entre train/val y test es una hipótesis importante para discutir;

\- que el audio suene distorsionado no implica fidelidad alta.



\---



**# 17. Experimento Starcaster 25**



Run:

\`starcaster25_cnn100_causal128_finealign_v1-\`



Resultados registrados:



\`\`\`text

current_epoch = 48

best_val_loss = 0.2586764097213745

best val epoch \~= 22



train loss inicial = 1.586309

train loss final   = 0.0965093



test_loss_final    = 0.7326936721801758

test_ESR_final     = 0.6173081398010254



test_loss_best     = 0.7216799855232239

test_ESR_best      = 0.6622310876846313



train_av_time      = 9.050043940544128 s

val_av_time        = 0.17006778717041016 s

\`\`\`



El campo \`total_time\` del JSON parece tener unidad o implementación poco clara y no debe usarse en el documento sin revisar el código.



**## 17.1 Memoria**



Archivo:

\`maxmemusage.txt\`



Valor:

\`3893308928\` bytes.



Conversión:

\- \~3.63 GiB de memoria CUDA máxima **\*\*asignada por PyTorch\*\*** durante el run / evaluación correspondiente.



No confundir:

\- memoria de entrenamiento;

\- memoria reservada;

\- memoria total de GPU;

\- memoria de inferencia.



Para la comparación final de eficiencia se debe medir inference peak memory con un protocolo común.



**## 17.2 Importancia del run 25**



Fue el primer run que verificó que el fix de memoria en \`dist_model.py\` permitía completar:

\- test del modelo final;

\- test del best model;

\- escritura de métricas;

sin OOM.



\---



**# 18. Starcaster 75 y 100 — estado al 2026-10-13**



**## 18.1 Captura**



75 y 100 fueron grabados con la misma cadena usada anteriormente.



**### Error del primer train 100**

El archivo tenía Warp activado en Ableton.



Síntomas:

\- duración 385.071 s vs dry 340 s;

\- correlación casi nula;

\- alignment incoherente.



Fix:

\- regrabar con Warp OFF;

\- reemplazar archivo en Git;

\- resolver conflicto \`modify/delete\` después de que otro clone hubiera borrado la versión mala.



**## 18.2 Estado del preprocessing**



**\*\*FINAL / congelado\*\*** para 75 y 100.



Coarse lags:

\`\`\`text

75 train  -73137

75 val    -97806

75 test   -110589



100 train -73136

100 val   -97806

100 test  -110589

\`\`\`



Causal delays:

\`\`\`text

75 train  153

75 val    135

75 test   133



100 train 153

100 val   135

100 test  132

\`\`\`



Las medianas residuales verificadas quedaron en \~128 samples para los seis pares.



**## 18.3 CNN Starcaster 75%**



Run:

`starcaster75_cnn100_causal128_finealign_v1`



Configuración:

- GatedConvNet;

- 9 layers por block;

- 2 blocks;

- 8 channels;

- kernel size 3;

- dilation growth 2;

- batch size 8;

- máximo 100 epochs;

- validation cada epoch;

- GPU CUDA.



Resultados registrados:



```text

current_epoch = 46



train loss inicial = 1.0320559740

train loss final   = 0.2301872373



best_val_loss = 0.46115535498

best val epoch ≈ 20



test_loss_best    = 0.9256879687

test_ESR_best     = 1.0965813398



test_loss_final   = 0.9450876713

test_ESR_final    = 1.0921847820



train_av_time = 8.2470549345 s

val_av_time   = 0.1618680954 s

```



[RESULTADO] El entrenamiento reduce de forma clara la pérdida de entrenamiento y obtiene una validation sustancialmente mejor que el baseline inicial del proyecto.



[RESULTADO] Existe una brecha fuerte entre validation y test.



[OBSERVACIÓN CUALITATIVA] `best_val_out.wav` presenta auditivamente una distorsión clara y convincente. Los outputs de test del modelo best y final suenan menos distorsionados que el ejemplo de validation.



[INTERPRETACIÓN] El modelo sí aprende una transformación de distorsión, pero generaliza peor sobre el split de test.



La diferencia de nivel de entrada de test (~3.5 dB más alto que train/val) permanece como hipótesis importante, dado que el pedal representa un sistema no lineal.



[DECISIÓN] Para comparación entre arquitecturas se conservará como modelo principal el checkpoint seleccionado por mejor validation, no un checkpoint escogido observando test.



**## 18.4 CNN Starcaster 100%**



Run:

`starcaster100_cnn100_causal128_finealign_v1`



Resultados registrados:



```text

current_epoch = 59



train loss inicial = 1.1681324244

train loss final   = 0.2211835980



best_val_loss = 0.3242235482

best val epoch ≈ 33



test_loss_best    = 0.9539803267

test_ESR_best     = 1.1381312609



test_loss_final   = 0.9826183319

test_ESR_final    = 1.1482921839



train_av_time = 10.6224439144 s

val_av_time   = 0.2117401361 s

```



[RESULTADO] El modelo de 100% obtiene una validation mejor que el modelo de 75%, por lo que la dificultad de validation no aumenta de forma monotónica con la cantidad de gain.



[RESULTADO] Sin embargo, la generalización a test continúa siendo pobre, con ESR superior a 1.



[OBSERVACIÓN CUALITATIVA] Se observa nuevamente una diferencia perceptual entre la salida de validation y las salidas de test: la primera reproduce de forma más evidente el carácter de distorsión esperado.



[DECISIÓN] `model_best` será el checkpoint representativo del experimento 100%, ya que fue seleccionado exclusivamente mediante validation y además obtiene mejores métricas de test que el modelo final.



**## 18.5 Resumen de CNN Starcaster finalizadas**



Los cuatro niveles no nulos de gain considerados hasta ahora tienen CNN final entrenada:



| Gain | Best validation loss | Test loss — best | Test ESR — best |

|---:|---:|---:|---:|

| 25% | 0.258676 | 0.721680 | 0.662231 |

| 50% | ~0.26149 | 0.931544 | 1.008393 |

| 75% | 0.461155 | 0.925688 | 1.096581 |

| 100% | 0.324224 | 0.953980 | 1.138131 |



[RESULTADO] El ESR de test empeora progresivamente entre los modelos 25%, 50%, 75% y 100%, aunque el validation loss no presenta una tendencia monotónica equivalente.



[HIPÓTESIS] La combinación de mayor no linealidad y el cambio de distribución de amplitud del split de test puede estar contribuyendo a esta degradación. Esta explicación debe evaluarse antes de presentarla como conclusión definitiva.



[RESULTADO] Con 25%, 50%, 75% y 100% entrenados, queda completada la primera ronda CNN del Fender Starcaster para los cuatro settings de gain actualmente utilizados.



\---



**# 19. Git: episodio de reemplazo del WAV 100 train**



Secuencia registrada:



1\. Raw 75/100 fueron subidos inicialmente.

2\. Commit remoto tras el push inicial llegó a \`10e9ee6\`.

3\. Se detectó que \`starcaster_100_target_44_train.wav\` tenía Warp.

4\. Desde otro clone se realizó un commit que borró el archivo malo; remoto avanzó a \`0d087f8...\`.

5\. En desktop se reemplazó el WAV y se hizo:

   \`\`\`cmd

   git add -f data\starcaster\train\starcaster_100_target_44_train.wav

   git commit -m "Add corrected Starcaster 100 train recording"

   \`\`\`

   generando localmente \`747d4d8\`.

6\. Push rechazado por non-fast-forward.

7\. \`git pull origin main\` produjo:

   \`\`\`text

   CONFLICT (modify/delete)

   \`\`\`

8\. Se resolvió conservando la versión correcta local:

   \`\`\`cmd

   git add -f data\starcaster\train\starcaster_100_target_44_train.wav

   git commit -m "Resolve conflict and keep corrected Starcaster 100 train recording"

   git push origin main

   \`\`\`

9\. La grabación corregida pasó posteriormente \`check_alignment.py\`.



Lección:

\- cuando dos clones modifican / eliminan el mismo WAV grande, actualizar antes de grabar commit;

\- en conflictos de datos, verificar cuál versión física es la correcta antes de resolver.



\---



**# 20. Comandos útiles actuales**



**## 20.1 Check alignment de varios settings / splits**



CMD:



\`\`\`cmd

for %S in (75 100) do @for %P in (train val test) do @python scripts\check_alignment.py data\starcaster\\%P\starcaster_input\_%P.wav data\starcaster\\%P\starcaster\_%S_target_44\_%P.wav --window 20 --max-lag-ms 5000

\`\`\`



**## 20.2 Procesar dataset**



\`\`\`cmd

python scripts\align_dataset.py

\`\`\`



**## 20.3 Check drift**



\`\`\`cmd

python scripts\check_drift.py

\`\`\`



**## 20.4 Entrenamiento CNN 50 de referencia**



\`\`\`cmd

cd external\Automated-GuitarAmpModelling

python dist_model.py --load_config "" --data_location "..\\..\data\processed\starcaster\50" --file_name "starcaster50" --device starcaster50_cnn100_causal128_finealign_v1 --model GatedConvNet --epochs 100 --validation_f 1 --batch_size 8 --cuda 1 --save_location "..\\..\results"

\`\`\`



**## 20.5 Entrenamientos en serie**



En Windows CMD se puede usar:



\`\`\`cmd

comando_75 && comando_100

\`\`\`



La segunda ejecución empieza solo si la primera termina con exit code exitoso.



No entrenar 75 y 100 simultáneamente en la misma RTX 3060: se prefiere ejecución secuencial para evitar contención de memoria GPU y mantener comparabilidad.



\---



**# 21. Estado de avance global**



**## Distorsión / Starcaster**



| Configuración | Raw | Alignment | Drift / causal | compare_splits | CNN |

|---|---:|---:|---:|---:|---:|

| 0% | ? | ? | ? | ? | ? |

| 25% | ✅ | ✅ | ✅ | ✅ | ✅ |

| 50% | ✅ | ✅ | ✅ | ✅ | ✅ |

| 75% | ✅ | ✅ | ✅ | ✅ | ✅ |

| 100% | ✅ | ✅ | ✅ | ✅ | ✅ |



Los cuatro settings no nulos de gain utilizados actualmente (25/50/75/100) tienen dataset procesado y CNN final entrenada.



**## Compresor / MXR Dyna Comp**



\| Configuración | Raw | Alignment | Preprocess | CNN |

\|---|---:|---:|---:|---:|

\| 0% | ⏳ | ⏳ | ⏳ | ⏳ |

\| 25% | ⏳ | ⏳ | ⏳ | ⏳ |

\| 50% | ⏳ | ⏳ | ⏳ | ⏳ |

\| 75% | ⏳ | ⏳ | ⏳ | ⏳ |

\| 100% | ⏳ | ⏳ | ⏳ | ⏳ |



**## Arquitecturas**



| Arquitectura | Estado |

|---|---|

| CNN | primera ronda Starcaster 25/50/75/100 completada |

| LSTM | pendiente |

| TCN | pendiente |



\---



**# 22. Diferencia entre cronograma original y ejecución real**



Cronograma de propuesta:



\`\`\`text

Semanas 4–5: grabación

Semana 6: preprocessing

Semanas 7–8: CNN

Semana 9: LSTM

Semana 10: TCN

\`\`\`



La ejecución real se atrasó principalmente por:

\- construcción del dataset;

\- configuración de captura;

\- diferencias de formato;

\- identificación tardía de desalineación temporal;

\- investigación de causalidad;

\- debugging del repo Wright;

\- OOM de evaluación;

\- necesidad de regrabar datos;

\- validación de niveles y splits.



Este atraso debe explicarse como parte de la evolución metodológica, no ocultarse.



\---



**# 23. Reuniones y comunicación con asesora**



**## 23.1 Propuesta y correcciones**



Se revisaron:

\- objetivos;

\- metodología;

\- alcance;

\- cronograma;

\- recursos;

\- bibliografía.



Se dejó explícito:

\- comparación CNN/TCN/LSTM;

\- fidelidad + eficiencia;

\- distorsión + compresión;

\- dataset propio;

\- uso de implementaciones existentes adaptadas.



**## 23.2 Carta / ética**



La carta de aval del Comité de Ética fue enviada y firmada.



**## 23.3 Audios de ejemplo**



Se enviaron a la asesora ejemplos de:

\- entrada;

\- salida procesada.



**## 23.4 Reuniones de seguimiento**



Horario habitual reciente:

\- viernes \~12:30 PM.



Se reportó atraso respecto a CNN por construcción / depuración del dataset.



Para una reunión de avance, el relato técnico consolidado es:



1\. se validó el stack con Big Muff;

2\. se descubrió que las primeras capturas Starcaster requerían corrección de alineación;

3\. se implementó coarse alignment + criterio causal;

4\. se obtuvo una mejora sustancial de validation en Starcaster 50;

5\. se completó 25 con evaluación final;

6\. se detectó distribución de amplitud distinta en test;

7\. se completó preprocessing de 75/100;

8\. quedan compresor, TCN y LSTM.



\---



**# 24. Material directamente reutilizable para metodología**



**## 24.1 Adquisición**



\> Las señales fueron reproducidas desde una interfaz Native Instruments Komplete Audio 6 hacia el pedal físico y la salida procesada fue capturada nuevamente mediante la misma interfaz. Para cada configuración se utilizó el mismo archivo de entrada limpia, manteniendo constantes los controles secundarios y modificando únicamente el parámetro principal del efecto. Las grabaciones se realizaron a 44.1 kHz.



**## 24.2 Alineación**



\> Debido a la latencia introducida por la ruta de reproducción y captura, las señales de entrada y objetivo fueron alineadas mediante correlación cruzada. Inicialmente se estimó un desplazamiento grueso por par de archivos. Posteriormente se introdujo un margen causal y se evaluó el desfase residual en cinco ventanas distribuidas a lo largo de cada grabación. El retardo adicional se ajustó para que la mediana del desfase residual fuera aproximadamente 128 muestras, equivalentes a 2.90 ms a 44.1 kHz.



**## 24.3 Polaridad**



\> Las señales capturadas presentaron una correlación negativa consistente con una inversión de polaridad en la cadena física. Esta inversión no fue corregida artificialmente, dado que forma parte de la transformación observada del sistema que el modelo debe aproximar.



**## 24.4 Formato**



\> Después de la alineación, las señales de entrada y objetivo fueron convertidas a PCM de 16 bits y organizadas en particiones de entrenamiento, validación y prueba compatibles con la implementación seleccionada.



**## 24.5 Comparabilidad**



\> Para comparar las arquitecturas se mantendrán las mismas divisiones de datos y se realizarán las mediciones de eficiencia en un entorno de hardware común. La fidelidad será evaluada mediante ESR y métricas temporales y espectrales, mientras que el costo computacional se caracterizará mediante número de parámetros, tamaño del modelo, tiempo de inferencia y factor de tiempo real.



\---



**# 25. Material directamente reutilizable para resultados preliminares**



**## 25.1 Validación del pipeline**



El GatedConvNet logró un best validation loss cercano a \`0.1012\` con el dataset Big Muff de referencia, demostrando que la implementación puede aprender una distorsión bajo datos correctamente preparados.



**## 25.2 Efecto de la alineación en Starcaster 50**



El run inicial de 20 epochs obtuvo best val \~\`0.704\` y produjo una salida casi silenciosa.



Después de:

\- coarse alignment;

\- margen causal;

\- fine alignment;



el best val mejoró a \~\`0.2615\`.



Esto constituye evidencia preliminar de que la sincronización de los pares input/target es una variable crítica en este problema.



**## 25.3 Generalización a test**



Starcaster 50:

\- best val \~0.2615;

\- test ESR \~1.0084.



Starcaster 25:

\- best val \~0.2587;

\- test ESR final \~0.6173;

\- test ESR best model \~0.6622.



Los resultados muestran una brecha importante entre validation y test.



Se identificó que el dry test tiene mayor RMS que train/val (\~3–4 dB), por lo que el cambio de nivel de entrada es una posible causa de degradación, especialmente en un sistema no lineal.



\---



**# 26. Problemas encontrados y lecciones**



**## 26.1 Dataset insuficientemente sincronizado**

**\*\*Síntoma:\*\*** CNN no aprendía bien Starcaster aunque Big Muff sí.  

**\*\*Lección:\*\*** verificar alineación temporal antes de entrenar modelos.



**## 26.2 Búsqueda de lag demasiado estrecha**

\`check_alignment.py\` inicialmente buscaba ±100 ms.



En 75/100 aparecieron resultados pegados al límite con correlación muy baja.



Al ampliar a ±5 s se encontraron offsets reales de \~1.66–2.51 s.



**\*\*Lección:\*\*** una correlación pegada al límite de búsqueda y con magnitud baja es una señal de que el search window es insuficiente.



**## 26.3 Warp accidental en Ableton**

**\*\*Síntoma:\*\*** train 100 duraba \~385 s frente a 340 s de dry.  

**\*\*Fix:\*\*** regrabar con Warp OFF.  

**\*\*Lección:\*\*** incluir Warp OFF en checklist obligatorio antes de cada captura.



**## 26.4 \`.numpy()\` sobre tensor con gradiente**

**\*\*Fix:\*\*** \`.detach().cpu().numpy()\`.



**## 26.5 OOM en evaluación final**

**\*\*Fix:\*\*** liberar red/tensores, vaciar cache, usar \`no_grad()\` y recargar best model.



**## 26.6 Git y archivos grandes**

**\*\*Síntoma:\*\*** warnings >50 MB y conflictos entre clones.  

**\*\*Lección:\*\*** tener estrategia de dataset / LFS para versión final y hacer pull antes de modificar archivos versionados desde otro equipo.



\---



**# 27. Checklist obligatorio para futuras grabaciones**



Antes de grabar cualquier configuración:



\`\`\`text

[ ] 44.1 kHz

[ ] Komplete Audio ASIO Driver

[ ] buffer definido

[ ] Driver Error Compensation = 0 ms

[ ] dry correcto cargado

[ ] WARP OFF

[ ] Clip Gain 0 dB

[ ] no plugins

[ ] salida física correcta

[ ] pedal correcto

[ ] control principal en posición registrada

[ ] controles secundarios fijos

[ ] input físico correcto

[ ] track wet armado

[ ] Monitor Off

[ ] duración esperada razonable

[ ] export nombre correcto

[ ] check_alignment antes de entrenar

\`\`\`



\---



**# 28. Próximos pasos inmediatos**



Estado al cierre de este log:



1\. parametrizar / ejecutar \`compare_splits.py\` para Starcaster 75 y 100;

2\. revisar especialmente RMS/peak y confirmar si el mismo test hotter aparece;

3\. lanzar CNN 75;

4\. lanzar CNN 100 inmediatamente después en serie;

5\. registrar stats/resultados;

6\. comenzar / completar adquisición del compresor;

7\. reutilizar pipeline genérico de alignment + drift;

8\. después continuar con LSTM;

9\. después TCN;

10\. estandarizar evaluación de fidelidad y eficiencia;

11\. comenzar ya el borrador del documento usando las secciones “Material reutilizable”.



\---



**# 29. Estrategia para mantener este diario**



Este archivo no debe depender de que el estudiante escriba manualmente después de cada sesión.



Flujo recomendado:



\`\`\`text

trabajo / terminal / experimento

        ↓

pegar output o archivos relevantes en ChatGPT

        ↓

ChatGPT interpreta y decide qué cambió

        ↓

actualizar este diario maestro

        ↓

commit del diario junto con el código/resultados relevantes

\`\`\`



Formato de futuras entradas:



\`\`\`markdown

**## YYYY-MM-DD — título corto**



**\*\*Tipo:\*\*** [EXPERIMENTO] [FIX] [RESULTADO] ...



**### Qué se hizo**

...



**### Comandos / configuración**

...



**### Resultado**

...



**### Interpretación**

...



**### Decisión**

...



**### Impacto en la tesis**

...



**### Siguiente paso**

...

\`\`\`



Regla:

\- no borrar un fracaso;

\- no ocultar un cambio de metodología;

\- registrar por qué se tomó una decisión;

\- separar dato observado de interpretación;

\- cuando una cifra vaya al documento final, indicar de qué archivo o run salió.



\---



**# 30. Automatización futura del log de experimentos**



El diario narrativo seguirá siendo mantenido por ChatGPT, porque un script no puede inferir correctamente decisiones metodológicas, reuniones o razones de diseño.



Sin embargo, conviene automatizar la parte cuantitativa.



Script futuro recomendado:

\`scripts/build_experiment_registry.py\`



Objetivo:

\- recorrer \`results/\`;

\- detectar \`training_stats.json\`;

\- leer nombre del run;

\- arquitectura;

\- epoch final;

\- best validation;

\- test loss;

\- test ESR;

\- tiempos;

\- \`maxmemusage.txt\`;

\- commit Git actual;

\- fecha;

\- escribir una tabla generada en \`docs/experiment_registry.md\`.



Así:

\- \`diario_tesis.md\` = historia, decisiones, problemas, metodología;

\- \`experiment_registry.md\` = métricas generadas automáticamente.



[PENDIENTE] Implementar después de lanzar 75/100 para no interrumpir el avance inmediato.



\---



**# 31. Pendientes de verificación antes del informe final**



\- confirmar estado real de Starcaster 0%;

\- confirmar configuración exacta / nombres de controles del pedal usados en cada sesión;

\- verificar número exacto de parámetros del GatedConvNet desde código;

\- verificar receptive field exacto;

\- verificar learning rate y cualquier scheduler del run final;

\- definir protocolo final de inference timing;

\- medir inference memory separada de training memory;

\- definir cómo reportar test hotter / OOD;

\- completar compresor;

\- ejecutar LSTM y TCN;

\- decidir almacenamiento definitivo de WAVs grandes;

\- convertir resultados relevantes en tablas y figuras;

\- sincronizar README con estructura final;

\- producir mini software / producto solicitado por la asesora.



\---



**# 32. Resumen ejecutivo de estado — 2026-10-13**



El proyecto ya cuenta con:



\- propuesta formal con compresión + distorsión;

\- protocolo de adquisición funcional;

\- repositorio propio;

\- fork / submódulo de Wright adaptado;

\- entorno PyTorch/CUDA funcional en RTX 3060;

\- scripts de alignment;

\- preprocessing reproducible;

\- causal alignment documentado;

\- Starcaster 25 CNN entrenada y evaluada;

\- Starcaster 50 CNN entrenada y evaluada;

\- Starcaster 75 CNN entrenada y evaluada;

\- Starcaster 100 CNN entrenada y evaluada;

\- control exitoso sobre Big Muff;

\- diagnóstico documentado de problemas de dataset y test distribution.



Lo que falta concentra el trabajo en:

\- capturar y procesar compresor;

\- revisar con la asesora la interpretación de la primera ronda CNN Starcaster;

\- entrenar LSTM/TCN;

\- evaluación comparativa final;

\- redacción.



Este archivo debe actualizarse en cada sesión significativa y utilizarse como base del documento de tesis.



\---



**# 33. Registro cronológico resumido**



**## 2026-08 — definición y corrección de propuesta**

[ALCANCE] Se consolidó el proyecto alrededor de **\*\*compresión y distorsión\*\***, comparando CNN, TCN y LSTM.  

[METODOLOGÍA] Se decidió usar un dataset propio de pares input/output y adaptar implementaciones existentes.  

[ASESORA] Se realizaron correcciones de objetivos, metodología, cronograma, recursos y bibliografía.  

[DECISIÓN] Se fijó el criterio de comparar fidelidad y eficiencia, no solo loss de entrenamiento.



**## 2026-08-14**

[ASESORA] Objetivos reportados como aprobados.  

[ALCANCE] Se mantiene comparación CNN/TCN/LSTM, dataset físico, métricas temporales/frecuenciales y costo computacional.



**## 2026-08-30 / semana 6**

[ASESORA] Reunión de seguimiento relevante; ya existía presión de cronograma por dataset y CNN.



**## 2026-09-01**

[DATASET] Trabajo sobre cadena de grabación con Komplete Audio 6 y Ableton.  

[METODOLOGÍA] Se empezó a formalizar captura clean -> pedal -> target.



**## 2026-09-16**

[IMPLEMENTACIÓN] Se clonó \`Automated-GuitarAmpModelling\` en desktop.  

[GIT] Se verificó CoreAudioML como submódulo.



**## 2026-09-17**

[IMPLEMENTACIÓN] Instalación de PyTorch.  

[PROBLEMA] Hubo congelamientos durante instalaciones pesadas.  

[EXPERIMENTO] Se probó demo Big Muff / Blackstar.



**## 2026-09-18**

[DATASET] Se discutió conversión WAV y sample rate 44.1/48 kHz.  

[DECISIÓN] El pipeline actual terminó consolidado en 44.1 kHz.



**## 2026-09-19**

[DATASET] Se midieron duraciones del dataset.  

[HARDWARE] Se confirmó conveniencia de usar laptop para entrenamiento.



**## 2026-09-20**

[GIT] Se creó / consolidó repo \`Tesis-Audio-DL\`.  

[GIT] Se organizó \`data/raw\`, \`data/processed\`, \`scripts\`, \`configs\`, \`experiments/cnn\`.  

[ASESORA] Se enviaron audios de ejemplo de entrada y salida.



**## 2026-09-24 / 2026-09-25**

[IMPLEMENTACIÓN] Se trabajó con Wright como submódulo.  

[EXPERIMENTO] Primer Starcaster 50 CNN 20 epochs.  

[PROBLEMA] Output casi silencioso y validation alta.  

[FIX] Se corrigió posteriormente \`.detach()\` para export final.  

[ASESORA] Había necesidad urgente de mostrar avance de CNN.



**## Finales de septiembre / principios de octubre**

[METODOLOGÍA] Se abandonó la idea de seguir entrenando a ciegas y se priorizó validar alineación del dataset.  

[IMPLEMENTACIÓN] Nacen \`check_alignment.py\`, \`align_dataset.py\`, \`check_drift.py\`, \`compare_splits.py\`.



**## 2026-10-07**

[PROBLEMA] Se confirmó que el round-trip físico podía introducir offsets que Wright no corregía.  

[DECISIÓN] Medir lag antes de cualquier nuevo entrenamiento.  

[METODOLOGÍA] Se formalizó coarse alignment y luego causal alignment.



**## 2026-10 — Starcaster 50 fine alignment**

[EXPERIMENTO] \`starcaster50_cnn100_causal128_finealign_v1-\`.  

[RESULTADO] Best val \~0.26149.  

[PROBLEMA] OOM en evaluación final.  

[FIX] Evaluación independiente + arreglo de memoria en \`dist_model.py\`.  

[RESULTADO] Test ESR \~1.00839.



**## 2026-10 — Starcaster 25**

[EXPERIMENTO] \`starcaster25_cnn100_causal128_finealign_v1-\`.  

[RESULTADO] Best val 0.2586764.  

[RESULTADO] Test ESR final 0.6173081.  

[RESULTADO] Run verificó fix de memoria.



**## 2026-10-13 — grabación 75/100**

[DATASET] Se grabaron Starcaster 75 y 100.  

[GIT] Se subieron WAVs raw.  

[PROBLEMA] Train100 tenía Warp y duración \~385 s.  

[FIX] Se regrabó y reemplazó.  

[RESULTADO] Train100 corregido: lag -73136, corr -0.4738.



**## 2026-10-13 — preprocessing 75/100**

[RESULTADO] Coarse alignment consistente entre 75/100.  

[METODOLOGÍA] Se midió drift con cinco ventanas.  

[DECISIÓN] Se ajustaron delays para centrar mediana residual en 128 samples.  

[RESULTADO] 75 delays finales 153/135/133.  

[RESULTADO] 100 delays finales 153/135/132.  

[DECISIÓN] Preprocessing 75/100 congelado.  



**## 2026-10-13 — cierre de primera ronda CNN Starcaster 75/100**



[PROBLEMA] `compare_splits.py` detectó que los targets nuevos de 75/100 habían sido exportados accidentalmente como stereo mientras los inputs eran mono.



[FIX] Se convirtieron los WAV afectados offline a mono conservando el canal 0. Se decidió no incorporar esta corrección a `align_dataset.py`, dado que correspondía a un error puntual de exportación y no al pipeline experimental deseado.



[RESULTADO] Después de la corrección, `align_dataset.py` y `compare_splits.py` completaron correctamente.



[RESULTADO] Se confirmó nuevamente el cambio de nivel del dry de test: aproximadamente -19.57 dBFS frente a ~-23.1/-23.4 dBFS en train/validation.



[EXPERIMENTO] Se entrenó `starcaster75_cnn100_causal128_finealign_v1`.



[RESULTADO] Starcaster75:

- current_epoch = 46;

- best val = 0.461155;

- test loss best = 0.925688;

- test ESR best = 1.096581;

- test loss final = 0.945088;

- test ESR final = 1.092185.



[OBSERVACIÓN] El output de validation presenta auditivamente una distorsión más evidente que los outputs de test.



[EXPERIMENTO] Se entrenó `starcaster100_cnn100_causal128_finealign_v1`.



[RESULTADO] Starcaster100:

- current_epoch = 59;

- best val = 0.324224;

- test loss best = 0.953980;

- test ESR best = 1.138131;

- test loss final = 0.982618;

- test ESR final = 1.148292.



[RESULTADO] Los cuatro settings 25/50/75/100 del Starcaster cuentan ahora con una CNN entrenada y evaluada.



[DECISIÓN] Para comparaciones futuras se utilizará el checkpoint seleccionado por mejor validation. El test se reserva para evaluación y no para selección de modelos.



[INTERPRETACIÓN] Los resultados sugieren que la red logra aprender el carácter de distorsión sobre train/validation, pero existe una brecha importante de generalización hacia test. El cambio de distribución de amplitud del dry de test es una hipótesis prioritaria que debe investigarse.



[PENDIENTE] Revisar con la asesora la interpretación de estos entrenamientos y ajustar los siguientes pasos antes de lanzar LSTM/TCN o comenzar una nueva fase de adquisición.



\---



**\*\*FIN DEL ESTADO RECONSTRUIDO HASTA 2026-10-13.\*\***
