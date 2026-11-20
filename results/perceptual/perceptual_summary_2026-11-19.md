# Evaluación perceptual exploratoria — corte del 19 de noviembre de 2026

**Estado:** corte de datos de doce cuestionarios, última respuesta registrada el 19-11-2026 a las 23:51:58 (hora de la hoja). **Fuente privada:** Google Sheets `Tesis - Respuestas evaluacion perceptual`, pestaña `Form Responses 1`. La primera respuesta fue completada por el propio investigador, según declaró, sin consultar la clave A/B/C. Se excluye del resumen principal y se muestra n=12 total como sensibilidad.

## Procedimiento ejecutado

- Reclutamiento de conveniencia mediante contactos personales e historia de Instagram.
- Dos efectos físicos: Fender Starcaster Distortion y MXR Dyna Comp, ambos en configuración 50 %.
- Interpretación externa no utilizada para entrenamiento, validación ni test original.
- Fragmento escogido manualmente de **2:40 a 3:05 (160–185 s, 25 s)** con idéntico intervalo para dry, pedal real y las tres predicciones de cada efecto.
- WAV originales convertidos a MP4 con imagen estática y audio AAC; videos de YouTube no listados incrustados en Google Forms. El mismo clip dry se reutiliza en ambos bloques.
- El oyente sabía cuál audio correspondía al dry y al pedal real; los otros se presentaron como modelos A/B/C sin revelar la arquitectura.
- Puntajes de fidelidad en escala **1–10**, selección múltiple de diferencias sonoras, comentarios abiertos opcionales y elección del modelo más fiel por efecto.
- No hubo normalización individual de loudness ni compensación temporal adicional de las predicciones. El orden A/B/C era fijo y no se contrabalanceó.

## Participantes

Se completaron **12 formularios** con consentimiento, incluido uno del investigador; el análisis principal corresponde a **11 evaluadores externos**. En total, cinco usaron audífonos y siete parlantes de computador o celular. La experiencia autodeclarada fue ninguna (7), básica/aficionado (3) e intermedia (2). Las condiciones y equipos de escucha no fueron controlados.

## Resultados descriptivos

| Efecto | Modelo | Arquitectura | Media externos (n=11) | Mediana [Q1–Q3] | Media total (n=12) | Preferido (n=11) |
|---|---|---|---:|---:|---:|---:|
| Starcaster | A | LSTM | 6.91 | 7 [6.00–8.50] | 7.00 | 1/11 |
| Starcaster | B | CNN | 7.82 | 8 [7.00–9.00] | 7.75 | 5/11 |
| Starcaster | C | TCN | 7.36 | 8 [6.50–8.50] | 7.33 | 4/11 |
| Dyna Comp | A | LSTM | 6.64 | 7 [5.00–8.00] | 6.42 | 1/11 |
| Dyna Comp | B | TCN | 3.27 | 3 [1.00–5.00] | 3.08 | 1/11 |
| Dyna Comp | C | CNN | 8.45 | 9 [7.50–9.50] | 8.00 | 9/11 |

En el **Starcaster**, CNN (Modelo B) recibió el mayor promedio externo de fidelidad (**7.82/10**), seguida de TCN (Modelo C, 7.36/10) y LSTM (Modelo A, 6.91/10). Cinco de los 11 externos seleccionaron CNN como más fiel, cuatro TCN, uno LSTM y uno indicó no percibir una diferencia clara.

En el **Dyna Comp**, CNN (Modelo C) obtuvo **8.45/10**, LSTM (Modelo A) 6.64/10 y TCN (Modelo B) 3.27/10. Nueve de los 11 externos seleccionaron CNN como más fiel, uno LSTM y uno TCN. Esto no coincide con el orden del ESR de la prueba externa en la grabación completa, donde LSTM obtuvo el menor ESR (1.048534). La discrepancia es descriptiva y no demuestra por sí sola qué propiedad acústica explica la preferencia.

## Interpretación y limitaciones

La muestra es reducida y de conveniencia; los resultados no pretenden inferencia poblacional. Solo se evaluó un único fragmento musical de 25 segundos por efecto y un setting del pedal (50 %); el orden de escucha fue fijo, hubo heterogeneidad de dispositivos y la compresión de YouTube (AAC y streaming) introduce una diferencia respecto de los WAV de origen. Al no igualar el volumen individualmente, los oyentes podían penalizar diferencias de nivel junto con otras características. Los comentarios abiertos describen volumen, timbre, ataque, sustain y artefactos, pero no bastan para atribuir causas específicas.

El par dry/pedal real del conjunto externo de Dyna Comp presentó +131 muestras frente a una meta de +128 de retardo residual. Esto **no equivale a verificar la alineación de cada predicción**. No se manipularon los audios de los modelos ni se reentrenó después de diseñar la escucha.

## Reproducibilidad

- Recorte: `python scripts/prepare_perceptual_eval.py --start 160 --duration 25`, semilla predeterminada 42.
- Correspondencias: Starcaster A=LSTM, B=CNN, C=TCN; Dyna Comp A=LSTM, B=TCN, C=CNN.
- Métricas objetivas complementarias: `results/generalization/generalization_summary.csv`.
- No se publican respuestas individuales ni datos personales; solo resultados agregados.
- **Corte dinámico:** si llegan respuestas posteriores al 19-11-2026 23:51:58, se debe volver a calcular el agregado y declarar el nuevo tamaño de muestra.
