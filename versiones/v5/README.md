# RoBERTa v5

Prueba corta con minería de errores de TRAIN.

## Archivos y cambios

Mezcla de ejemplos normales y errores difíciles; selección por mejora armónica sin empeorar EM de ninguna clase.

- `entrenar_v5_rapido.py`

Usa `project_paths.ROOT`; no depende del directorio actual. Los artefactos continúan en la raíz, sin mover datos, modelos, salidas ni logs. `qa_v3.py` es compartido; V4 usa su propio `qa_v4.py`. Las utilidades de umbral importan `medir` de `versiones.v1.evaluar`.

## Comandos desde la raíz (PowerShell)

```powershell
.\.venv\Scripts\python.exe versiones/v5/entrenar_v5_rapido.py
.\.venv\Scripts\python.exe versiones/v5/entrenar_v5_rapido.py --evaluar
```

El primer comando entrena; no se ejecutó durante esta reorganización. La evaluación escribe registros: revisa sus destinos antes de repetirla si deseas preservar una corrida. 

## Resultados y decisión

Seleccionó v5_candidate en seguimiento, pero no se adoptó en la API: en las 9873 preguntas subió EM global y sin respuesta, bajaron EM/F1 con respuesta, F1 global y media armónica frente a V2.

Fuente: `logs/roberta_squad2_v5_rapido/resumen.json`. Selección registrada: `v5_candidate`.

Candidato en las 1000 preguntas de seguimiento:

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 1000 | 79.5000 | 81.6606 |
| answerable | 486 | 73.8683 | 78.3139 |
| unanswerable | 514 | 84.8249 | 84.8249 |

Referencia V2 en las mismas preguntas:

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 1000 | 78.6000 | 80.9360 |
| answerable | 486 | 73.6626 | 78.4691 |
| unanswerable | 514 | 83.2685 | 83.2685 |

Registro posterior: `logs/roberta_squad2_v5_rapido/metricas_evaluacion.json`. 

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 9873 | 79.1249 | 82.0755 |
| answerable | 4949 | 73.4088 | 79.2951 |
| unanswerable | 4924 | 84.8700 | 84.8700 |

Las configuraciones, historiales y gráficas disponibles se conservan en `logs/roberta_squad2_v5_rapido/`. Los umbrales históricos alternativos no deben confundirse con `logs/roberta_squad2_v2/umbral.json`, que usa la API. Las particiones de desarrollo y las 9873 preguntas fueron reutilizadas; no son un test independiente. No se inventan métricas faltantes.
