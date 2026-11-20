# Protocolo de evaluación perceptual

**Estado:** ejecutado; corte exploratorio 19-11-2026, n=12 (once externos y una respuesta del investigador)  
**Fecha:** 2026-11-19

## Objetivo

Complementar las métricas objetivas con una evaluación perceptual y cualitativa de qué tan fielmente las salidas generadas por CNN, LSTM y TCN reproducen el efecto del pedal físico.

La prueba no busca que los participantes identifiquen arquitecturas ni que sepan cómo funciona una red neuronal. Tampoco se plantea principalmente como una prueba de preferencia musical. El participante conoce cuál audio es la guitarra limpia, cuál proviene del pedal físico real y cuáles corresponden a emulaciones computacionales. Su tarea es comparar cada emulación con el pedal real y describir qué tan fiel le parece.

## Material de audio

Se utilizará el mismo material externo empleado en la prueba adicional de generalización, correspondiente a una interpretación que no pertenecía a train, validation ni test del dataset original.

Para cada efecto se prepararán cinco archivos:

1. guitarra limpia, sin pedal;
2. salida del pedal físico real;
3. emulación mediante Modelo A;
4. emulación mediante Modelo B;
5. emulación mediante Modelo C.

Los cinco archivos utilizan exactamente el mismo intervalo temporal.

Para la evaluación ejecutada se eligió manualmente el intervalo de 2:40 a 3:05 (**25 s**), con `--start 160 --duration 25`, manteniendo exactamente el mismo intervalo para dry, pedal físico y los tres modelos de ambos efectos. La selección del fragmento no fue aleatoria.

El mismo intervalo se utiliza para Starcaster y Dyna Comp. El script realiza todos los recortes automáticamente, por lo que no se requiere edición manual de audio.

## Identificación de los audios

El participante conoce explícitamente:

- cuál archivo es la señal limpia;
- cuál archivo corresponde al pedal físico real;
- cuáles archivos son emulaciones producidas mediante modelos de aprendizaje profundo.

Las tres arquitecturas se presentan únicamente como Modelo A, Modelo B y Modelo C. El propósito de A/B/C no es pedir al participante que adivine la arquitectura. Se utiliza para evitar que términos técnicos como CNN, LSTM o TCN introduzcan expectativas innecesarias en la evaluación.

El mapeo A/B/C se genera de forma reproducible y queda guardado localmente en `answer_key.json`.

## Nivel de audio

No se aplica normalización independiente ni gain matching entre pedal real y predicciones.

Esta decisión es intencional porque la reproducción correcta del nivel hace parte de la fidelidad estudiada y ya aparece en la evaluación cuantitativa mediante error LUFS. Corregir artificialmente el volumen antes de la escucha ocultaría una diferencia real del modelo.

Se indicará al participante ajustar el volumen general a un nivel cómodo antes de comenzar y mantenerlo estable durante cada bloque.

## Estructura del formulario

### Introducción y consentimiento

Texto propuesto:

> Esta encuesta hace parte de una evaluación académica de modelos de audio para guitarra eléctrica. Escucharás primero una señal de guitarra limpia, después la salida producida por un pedal físico real y finalmente tres emulaciones computacionales del mismo efecto. Tu tarea es comparar cada emulación con el pedal real y evaluar qué tan fiel te parece. No necesitas conocer cómo funcionan los modelos. La participación es voluntaria y las respuestas se analizarán de forma agregada. Se recomienda utilizar audífonos o parlantes con los que puedas escuchar cómodamente y ajustar el volumen antes de comenzar.

Pregunta obligatoria:

**¿Aceptas participar voluntariamente en esta evaluación?**

- Sí.
- No.

Si la respuesta es No, el formulario debe finalizar.

### Experiencia del participante

Pregunta opcional de selección única:

**¿Cómo describirías tu experiencia con guitarra, producción musical o audio?**

- Ninguna o muy poca.
- Básica.
- Intermedia.
- Avanzada o profesional.

Esta variable se utilizará solo de forma descriptiva salvo que el número de participantes permita una comparación razonable entre grupos.

## Bloque 1. Fender Starcaster Distortion

Presentar en este orden:

1. **Guitarra limpia, sin pedal.**
2. **Fender Starcaster Distortion real.**
3. **Modelo A.**
4. **Modelo B.**
5. **Modelo C.**

Para cada modelo A/B/C se repiten las siguientes preguntas.

### Pregunta cuantitativa principal

**Comparando el Modelo X con el pedal real, ¿qué tan fiel te parece la emulación del efecto?**

Escala de 1 a 10:

- 1 = Nada fiel.
- 10 = Prácticamente indistinguible del pedal real.
- Valores intermedios = grados crecientes de fidelidad.

### Pregunta de selección múltiple

**¿Qué diferencias notas entre el Modelo X y el pedal real? Selecciona todas las que apliquen.**

- Nivel o volumen.
- Timbre o color del sonido.
- Cantidad o carácter de la distorsión.
- Ataque o respuesta al tocar.
- Sustain.
- Ruido o artefactos.
- No noto una diferencia clara.
- Otro.

### Pregunta abierta

**Si quieres, describe con tus propias palabras cualquier diferencia que notes entre el Modelo X y el pedal real.**

Respuesta abierta opcional.

### Comparación final del bloque

**De los Modelos A, B y C, ¿cuál te pareció más fiel al pedal real?**

- Modelo A.
- Modelo B.
- Modelo C.
- No percibí una diferencia clara.

Esta pregunta es complementaria. La medida principal sigue siendo la calificación individual de fidelidad de cada modelo.

## Bloque 2. MXR Dyna Comp

Presentar en este orden:

1. **Guitarra limpia, sin pedal.**
2. **MXR Dyna Comp real.**
3. **Modelo A.**
4. **Modelo B.**
5. **Modelo C.**

Para cada modelo se utiliza la misma escala de fidelidad de 1 a 10.

### Pregunta de selección múltiple

**¿Qué diferencias notas entre el Modelo X y el pedal real? Selecciona todas las que apliquen.**

- Nivel o volumen.
- Timbre o color del sonido.
- Ataque.
- Sustain.
- Cantidad o carácter de la compresión.
- Respuesta dinámica.
- Ruido o artefactos.
- No noto una diferencia clara.
- Otro.

### Pregunta abierta

**Si quieres, describe con tus propias palabras cualquier diferencia que notes entre el Modelo X y el pedal real.**

Respuesta abierta opcional.

Al final se repite la pregunta de cuál de los tres modelos pareció más fiel al pedal real.

## Instrucciones de escucha

- utilizar audífonos o parlantes en condiciones cómodas;
- ajustar el volumen antes de comenzar;
- mantener el volumen estable durante cada bloque;
- se permite repetir los audios;
- comparar fidelidad respecto al pedal real, no únicamente cuál sonido se prefiere;
- no es necesario conocer conceptos de redes neuronales.

## Análisis previsto

La evaluación combinó información cuantitativa y cualitativa. Los resultados del primer corte se documentan en `results/perceptual/perceptual_summary_2026-11-19.md`; el análisis principal excluye la respuesta del investigador.

Para cada efecto y arquitectura se reportará:

- número de respuestas válidas;
- mediana de fidelidad en la escala 1 a 10;
- rango intercuartílico;
- distribución de respuestas;
- frecuencia de categorías perceptuales seleccionadas;
- número y porcentaje de veces que la arquitectura fue seleccionada como la más fiel.

Las respuestas abiertas se resumirán por temas recurrentes, por ejemplo diferencias de nivel, timbre, distorsión, ataque, sustain o dinámica. No se asignarán categorías a priori que no aparezcan en las respuestas y no se presentarán comentarios individuales como representativos de toda la muestra.

Si se recoge experiencia del participante, se reportará como contexto descriptivo. No se realizarán afirmaciones inferenciales fuertes entre subgrupos si el tamaño de muestra no lo permite.

## Alcance y cautelas

La muestra fue de conveniencia y la evaluación utiliza un único fragmento por efecto al 50 %. Por lo tanto, sus resultados complementan las métricas objetivas pero no representan una evaluación perceptual exhaustiva de todos los settings ni de toda la población de oyentes.

La evaluación debe interpretarse junto con ESR, MAE, MR-STFT, error LUFS, la prueba externa de generalización y el benchmark de eficiencia.

## Ejecución documentada

Los WAV se convirtieron en nueve videos MP4 con imagen estática y audio AAC, subidos a YouTube como videos no listados e insertados en Google Forms. El dry es idéntico para ambos efectos y se reutilizó. Los participantes escucharon en condiciones no controladas; el orden A/B/C fue fijo. Las calificaciones reflejan fidelidad percibida bajo esas condiciones y no una prueba ABX o de laboratorio. Corte actualizado: doce respuestas completas (once externas en el análisis principal), con última respuesta registrada el 19-11-2026 a las 23:51:58.
