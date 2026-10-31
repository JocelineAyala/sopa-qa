# Sopa: experimento RoBERTa V3

Los scripts de este experimento están en `versiones/v3/`; `qa_v3.py` permanece en la raíz como módulo compartido:

- `entrenar_v3.py`
- `qa_v3.py`
- `evaluar_v3.py`
- `README_V3.md`

## Qué cambia

Se continúa desde los pesos de **nuestra V2** (`outputs/roberta_squad2_v2/modelo`), no desde un checkpoint SQuAD de terceros. El checkpoint base original sigue siendo `FacebookAI/roberta-base`.

- Hasta una época adicional sobre el conjunto de entrenamiento SQuAD 2.0.
- Batch físico 2, acumulación 8: batch efectivo 16 en una GPU.
- Learning rate máximo `3e-6`, calentamiento durante el 5 % de los pasos.
- Precisión mixta y gradient checkpointing, como en V2.
- Evaluación y guardado cada 500 pasos.
- Parada anticipada tras cuatro evaluaciones sin mejora de la métrica QA.

La validación usa los mismos primeros 2000 ejemplos de `validation.shuffle(seed=42)`, ahora separados:

1. Primeros 1000: calibrar el umbral de no respuesta para cada checkpoint.
2. Siguientes 1000: seleccionar checkpoints por la media armónica entre EM de preguntas con respuesta y EM de preguntas sin respuesta.

La media armónica favorece que ambas clases funcionen bien. No es la nota del docente. Se registran también EM, F1, pérdida, umbral y resultados separados por clase.

Las etiquetas de validación no se usan para actualizar pesos. Las preguntas públicas y las doce preguntas de diagnóstico no se incorporan al entrenamiento. Como estas particiones ya participaron en experimentos anteriores, no constituyen un test nuevo e independiente. Las otras 9873 preguntas se usan para comparación posterior y tampoco son nuevas: ya fueron evaluadas antes. El test oculto del docente sigue siendo la prueba externa.

## Primero: comprobar sin entrenar

En PowerShell, dentro de `C:\proyecto-qa`:

```powershell
.\.venv\Scripts\python.exe versiones/v3/entrenar_v3.py --comprobar
```

Esta comprobación carga nuestra V2, prepara los datos, mide su desempeño y verifica la integración con Trainer. No actualiza pesos. Debe terminar con `COMPROBACION CORRECTA`.

Las librerías ya utilizadas en V2 son suficientes. No reinstales PyTorch: conservá la versión con CUDA.

## Entrenar

```powershell
.\.venv\Scripts\python.exe versiones/v3/entrenar_v3.py
```

El script repite la comprobación inicial antes de entrenar. Al reanudar, recupera el último checkpoint completo de V3; el optimizador y su calendario son nuevos al continuar por primera vez desde V2.

No lances dos entrenamientos simultáneos. Dejá la laptop conectada y ventilada. Podés detener con Ctrl+C una vez. Para reanudar, usá el mismo comando. La exportación final solo ocurre cuando Trainer termina normalmente, incluida una parada anticipada.

## Selección conservadora al terminar

El candidato elegido por Trainer se exporta a `outputs/roberta_squad2_v3/candidato`.

Se mide nuevamente en precisión completa y se compara con V2, usando el umbral vigente de V2 y exactamente las mismas 1000 preguntas de seguimiento. El candidato se promueve solo si:

- mejora la media armónica de EM;
- no empeora EM en las preguntas con respuesta;
- no empeora EM en las preguntas sin respuesta.

Si no cumple las tres condiciones, `outputs/roberta_squad2_v3/modelo` contiene una copia de V2. El candidato V3 se conserva por separado. Ninguna versión original se sobrescribe y la API existente no cambia.

El archivo `logs/roberta_squad2_v3/resumen.json` indica `selected: v3_candidate` o `selected: parent_v2`. El umbral correspondiente queda en `logs/roberta_squad2_v3/umbral.json`.

No se garantiza una mejora. Puede haber un candidato que mejore una clase y empeore otra; se informa ese compromiso y se conserva V2.

## Comparación posterior

```powershell
.\.venv\Scripts\python.exe versiones/v3/evaluar_v3.py
```

Evalúa el modelo seleccionado en las otras 9873 preguntas, sin volver a ajustar el umbral. Compartí el resumen de ambas clases antes de cambiar la API. Aunque la selección de desarrollo favorezca V3, la comparación posterior puede no mejorar.

## Archivos para el reporte

En `logs/roberta_squad2_v3`:

- `configuracion.json`: hiperparámetros y procedencia.
- `particiones.json`: IDs de calibración y seguimiento.
- `baseline_v2.json`: referencia anterior al entrenamiento.
- `calidad_por_checkpoint.jsonl`: resultados de cada evaluación.
- `historial.json`, `loss.png`, `calidad.png`: registros y curvas.
- `resumen.json`: comparación y selección final.
- `umbral.json`, `umbral_candidato.json`: configuraciones seleccionada y candidata.
- `metricas_evaluacion.json`: comparación posterior, tras ejecutar el evaluador.

Conservá el historial Git y excluí `outputs/` del repositorio. No cambies la API hasta revisar los resultados.

## Validación realizada en el entorno de preparación

Se verificaron la sintaxis, la selección de umbrales frente a una búsqueda exhaustiva, las métricas por clase, el postprocesamiento con ventanas y tokens sintéticos, las posiciones de caracteres y la separación de los ejemplos de desarrollo. No se ejecutó entrenamiento real ni inferencia con los pesos de la laptop en este entorno; la comprobación inicial valida esa integración en tu instalación.
