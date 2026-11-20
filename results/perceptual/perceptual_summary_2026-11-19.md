# Evaluación perceptual exploratoria — corte del 19-11-2026

**Estado:** resultados preliminares. **Fuente:** hoja privada "Tesis - Respuestas evaluacion perceptual", pestaña "Form Responses 1". Nueve cuestionarios completos al cierre de este corte. La primera respuesta corresponde al investigador según su reporte; se conservan los nueve datos para sensibilidad, pero se analizan ocho participantes externos como muestra principal.

## Procedimiento

- Muestreo por conveniencia, principalmente contactos personales e historia de Instagram.
- Dos efectos: Fender Starcaster Distortion y MXR Dyna Comp, setting 50 %.
- Interpretación de guitarra ajena a train, validation y test originales.
- Fragmento manual seleccionado de 2:40 a 3:05 (160–185 s, **25 s**) idéntico entre dry, pedal real y emulaciones.
- WAV convertidos a MP4 con imagen estática y audio AAC, publicados en YouTube como no listados e insertados en Google Forms. El mismo dry se reutilizó para ambos efectos. YouTube introduce compresión adicional, por lo que la escucha no equivale exactamente a los archivos PCM originales.
- Pedal físico y dry identificados explícitamente; arquitecturas ocultas como A/B/C.
- Calificación de fidelidad de **1 a 10**, diferencias perceptuales, comentario abierto opcional y modelo más fiel por efecto.
- No se igualó el volumen de los modelos de forma independiente; el orden A/B/C fue fijo. La codificación de YouTube y el dispositivo del oyente son fuentes de variación no controladas.

## Participantes

Nueve formularios completos; uno fue respondido por el propio investigador sin consultar la clave, según informó. El análisis principal usa n=8 externos; se incluye n=9 a modo de sensibilidad. De los nueve: 4 escucharon con audífonos y 5 con parlantes de computador/celular; experiencia en guitarra o producción de audio: 6 ninguna, 1 básica y 2 intermedia.

## Resultados descriptivos

| Efecto | Modelo | Arquitectura (clave analítica) | Media n=8 (externos) | Media n=9 (total) | Preferido n=8 |
|---|---|---|---:|---:|---:|
| Starcaster | A | LSTM | 7.00 | 7.11 | 1/8 |
| Starcaster | B | CNN | 7.75 | 7.67 | 4/8 |
| Starcaster | C | TCN | 7.13 | 7.11 | 3/8 |
| Dyna Comp | A | LSTM | 7.13 | 6.78 | 0/8 |
| Dyna Comp | B | TCN | 3.63 | 3.33 | 1/8 |
| Dyna Comp | C | CNN | 8.63 | 8.00 | 7/8 |

La CNN obtiene la mayor valoración media entre participantes externos en ambos efectos, con diferencias pequeñas en Starcaster y una mayor separación en Dyna Comp. En Dyna Comp, CNN fue seleccionada como la emulación más fiel por 7 de los 8 externos. Este resultado no coincide con el menor ESR de la prueba externa, obtenido por LSTM, lo que refuerza la interpretación multidimensional de la fidelidad.

## Limitaciones y cautelas

Resultados exploratorios sin inferencia poblacional: n pequeño, muestreo por conveniencia, un único fragmento y setting por efecto, orden fijo de escucha, equipo de reproducción heterogéneo, audio comprimido por YouTube y ausencia de igualación individual de loudness. No se atribuyen las diferencias a una única causa. Los WAV y checkpoints originales no fueron modificados para la encuesta.

La alineación dry/pedal real de Dyna Comp fue de +131 muestras respecto a +128 esperadas. Esta comprobación no verifica por sí sola la alineación temporal de cada predicción. Las métricas objetivas y los puntajes perceptuales se reportan por separado.

## Reproducibilidad

- Recorte WAV: `python scripts/prepare_perceptual_eval.py --start 160 --duration 25`, seed predeterminada 42.
- Clave para análisis: Starcaster A=LSTM, B=CNN, C=TCN; Dyna Comp A=LSTM, B=TCN, C=CNN.
- Evaluación cuantitativa externa: `results/generalization/generalization_summary.csv`.
- No se publican respuestas individuales en el repositorio. Este es un **corte n=9** y debe actualizarse si se suman participantes.
