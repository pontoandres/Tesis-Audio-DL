# Tesis-Audio-DL

**Proyecto de grado — Universidad de los Andes**  
**Modelado de efectos de distorsión de guitarra eléctrica mediante técnicas de aprendizaje profundo**

## Descripción y objetivo

El proyecto busca modelar la respuesta de un pedal de distorsión de guitarra eléctrica a partir de pares de señales de audio: una entrada limpia (*dry*) y su correspondiente salida procesada (*wet*). Se plantea implementar y comparar arquitecturas **CNN, TCN y LSTM** en términos de fidelidad de audio y eficiencia computacional.

El dispositivo de referencia es un **Fender Starcaster Distortion** y la adquisición se realiza mediante una interfaz **Native Instruments Komplete Audio 6**. Las señales provienen de una misma interpretación enviada al pedal desde Ableton, de modo que el procesamiento y la comparación requieren verificar su alineación temporal.

## Guía rápida del repositorio

| Ruta | Contenido |
| --- | --- |
| `scripts/prepare50.py` | Preparación de los pares de audio del Starcaster al 50 %: conversión a PCM de 16 bits, comprobaciones y preparación de los archivos para el entrenador. La versión actual incorpora la revisión de la alineación temporal. |
| `external/Automated-GuitarAmpModelling/` | Repositorio de referencia de Alec Wright, incorporado como **submódulo Git**. Contiene la arquitectura CNN utilizada inicialmente y su código de entrenamiento. |
| `external/Automated-GuitarAmpModelling/dist_model.py` | Punto de entrada para configurar, entrenar y evaluar el modelo de distorsión. |
| `external/Automated-GuitarAmpModelling/CoreAudioML/networks.py` | Implementación de las redes, incluida **GatedConvNet**, una arquitectura neuronal convolucional con compuertas (CNN). |
| `results/starcaster50_cnn20-/` | Resultados conservados del **primer experimento** con el pedal al 50 % y 20 épocas de entrenamiento. |
| `data/starcaster/` | Organización local del conjunto de datos por particiones `train`, `val` y `test`. Los audios no tienen que estar necesariamente publicados en GitHub. |

### Primer experimento: GatedConvNet (CNN)

La arquitectura del primer experimento es **GatedConvNet**, tomada de la implementación de referencia de Wright; **no es una CNN desarrollada desde cero en este proyecto**. El trabajo propio incluye la preparación de los datos del pedal, la configuración del experimento y el análisis de los resultados.

Para consultar los resultados publicados, abrir [`results/starcaster50_cnn20-/`](results/starcaster50_cnn20-/):

| Archivo | Descripción |
| --- | --- |
| `model.json` | Arquitectura y parámetros guardados al finalizar las 20 épocas. |
| `model_best.json` | Arquitectura y parámetros de la versión con menor pérdida de validación. |
| `training_stats.json` | Historial de pérdidas y estadísticas registradas durante el entrenamiento. |
| `best_val_out.wav` | Audio generado por el modelo durante la validación. |

**Estado de este experimento:** el entrenamiento completó las 20 épocas, pero el resultado es **preliminar y no representa todavía una emulación satisfactoria**: las pérdidas se estancaron y la amplitud del audio generado es considerablemente menor que la salida del pedal. La exportación final de test falló por un error al convertir un tensor de PyTorch a NumPy; por eso no se presentan archivos ni métricas finales de test como resultados concluidos. Actualmente se está revisando la alineación temporal de los pares de audio y el procedimiento de evaluación antes de repetir el experimento.

## Cómo obtener el código de referencia

El repositorio de Wright está incluido como submódulo, no como una copia independiente de sus archivos. Para clonar este proyecto con el código de referencia:

```bash
git clone --recurse-submodules https://github.com/pontoandres/Tesis-Audio-DL.git
```

Si el proyecto ya fue clonado sin submódulos, ejecutar desde su raíz:

```bash
git submodule update --init --recursive
```

En GitHub, la carpeta `external/Automated-GuitarAmpModelling/` enlaza al repositorio original y a la revisión registrada por este proyecto.

## Tecnologías y equipo

- Python y PyTorch.
- Procesamiento digital de audio y aprendizaje profundo.
- Ableton Live Lite para la adquisición y el enrutamiento de audio.
- Native Instruments Komplete Audio 6 y Fender Starcaster Distortion.

## Trabajo en curso

1. Verificar la sincronización de las señales limpias y procesadas en las particiones de entrenamiento, validación y prueba.
2. Corregir la exportación del audio de prueba y evaluar la fidelidad frente al pedal real.
3. Continuar con la implementación y comparación de CNN, TCN y LSTM, incluyendo métricas de eficiencia computacional.

## Referencias de implementación

- [Automated-GuitarAmpModelling](https://github.com/Alec-Wright/Automated-GuitarAmpModelling) — base utilizada para la primera CNN.
- [NablAFx](https://github.com/mcomunita/nablafx) — referencia para modelado neuronal de efectos de audio.
