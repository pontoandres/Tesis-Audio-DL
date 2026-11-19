# Protocolo de evaluación perceptual

**Estado:** protocolo definido para ejecución  
**Fecha:** 2026-11-19

## Objetivo

Complementar las métricas objetivas con una evaluación perceptual sencilla de similitud entre la salida real de cada pedal y las predicciones de CNN, LSTM y TCN.

La prueba no busca medir preferencia musical. La pregunta central es qué tan similar se percibe cada emulación respecto a la salida real del dispositivo físico.

## Material de audio

Se utilizará el mismo material externo empleado en la prueba adicional de generalización, correspondiente a una interpretación que no pertenecía a train, validation ni test del dataset original.

Para cada efecto se preparará:

- una referencia producida por el pedal físico;
- una predicción CNN;
- una predicción LSTM;
- una predicción TCN.

Los cuatro archivos de cada efecto utilizarán exactamente el mismo intervalo temporal.

El script `scripts/prepare_perceptual_eval.py` selecciona por defecto un fragmento de 15 s utilizando únicamente la señal dry. La selección automática favorece una ventana activa con variación dinámica y no inspecciona las predicciones para decidir el fragmento. Esto reduce el riesgo de escoger una región porque favorezca o perjudique a una arquitectura particular.

## Anonimización

Las arquitecturas se presentan como A, B y C. El mapeo se genera por separado para cada efecto mediante una semilla reproducible y se guarda únicamente en un archivo local `answer_key.json`.

El answer key no debe publicarse ni compartirse con los participantes antes de cerrar la recolección de respuestas.

## Nivel de audio

No se aplica normalización independiente ni gain matching entre la referencia y las predicciones.

Esta decisión es intencional porque la correspondencia de nivel forma parte de la fidelidad evaluada en el proyecto y ya se mide cuantitativamente mediante error LUFS. Igualar el loudness de cada modelo eliminaría una diferencia real producida por la emulación.

A los participantes se les indicará ajustar el volumen general de reproducción a un nivel cómodo antes de comenzar y evitar modificarlo entre archivos de un mismo bloque.

## Estructura del formulario

### Introducción y consentimiento

Texto propuesto:

> Esta encuesta hace parte de una evaluación académica de modelos de audio para guitarra eléctrica. Escucharás fragmentos cortos producidos por un pedal real y por tres modelos computacionales. Tu tarea es calificar qué tan similares suenan las emulaciones respecto a la referencia. La participación es voluntaria y las respuestas se analizarán de forma agregada. No es necesario tener experiencia musical. Se recomienda usar audífonos o parlantes con los que puedas escuchar cómodamente y ajustar el volumen antes de comenzar.

Pregunta obligatoria:

**¿Aceptas participar voluntariamente en esta evaluación?**

- Sí.
- No.

Si la respuesta es No, el formulario debe finalizar.

### Experiencia del participante

Pregunta opcional:

**¿Cómo describirías tu experiencia con guitarra, producción musical o audio?**

- Ninguna o muy poca.
- Básica.
- Intermedia.
- Avanzada o profesional.

Esta variable solo se utilizará de forma descriptiva. No se debe forzar una comparación por grupos si el número de respuestas es pequeño.

## Bloque 1: Starcaster Distortion

Presentar primero **Starcaster REFERENCE**.

Después presentar A, B y C de ese efecto.

Para cada candidato:

**Comparando con la referencia, ¿qué tan similar te parece el audio A/B/C?**

Escala de 1 a 5:

1. Muy diferente.
2. Bastante diferente.
3. Moderadamente similar.
4. Muy similar.
5. Prácticamente indistinguible.

Al final del bloque:

**¿Cuál de los tres audios te pareció más similar a la referencia?**

- A.
- B.
- C.
- No percibí una diferencia clara entre ellos.

## Bloque 2: MXR Dyna Comp

Repetir exactamente la misma estructura utilizando **Dyna Comp REFERENCE** y los candidatos A, B y C asignados para ese efecto.

## Instrucciones de escucha

- utilizar audífonos o parlantes en condiciones cómodas;
- ajustar el volumen antes de comenzar;
- no modificar deliberadamente el volumen entre referencia y candidatos;
- se permite repetir los audios;
- responder según similitud percibida, no según cuál sonido se prefiere.

## Análisis previsto

La prueba se analizará de forma descriptiva.

Para cada efecto y arquitectura se reportará:

- número de respuestas válidas;
- mediana de similitud en la escala 1 a 5;
- rango intercuartílico;
- distribución de respuestas;
- número y porcentaje de veces que la arquitectura fue seleccionada como la más similar.

Si se recoge experiencia del participante, se puede mostrar como contexto descriptivo. No se realizarán afirmaciones inferenciales fuertes entre subgrupos si el tamaño de muestra no lo permite.

## Alcance y cautelas

La muestra será de conveniencia y la evaluación utiliza un único fragmento por efecto al 50 %. Por lo tanto, sus resultados complementan las métricas objetivas pero no representan una prueba perceptual exhaustiva de todos los settings ni de toda la población de oyentes.

La evaluación perceptual debe interpretarse junto con ESR, MAE, MR-STFT, error LUFS, la prueba externa de generalización y el benchmark de eficiencia.
