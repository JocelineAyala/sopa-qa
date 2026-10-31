# RoBERTa v3

Continuación conservadora desde V2.

## Archivos y cambios

Learning rate 3e-6, máximo una época adicional; 1000 preguntas de calibración y 1000 de seguimiento.

- `entrenar_v3.py`
- `evaluar_v3.py`
- `README_V3.md` — documentación original preservada; sus instrucciones de ubicación fueron actualizadas.

Usa `project_paths.ROOT`; no depende del directorio actual. Los artefactos continúan en la raíz, sin mover datos, modelos, salidas ni logs. `qa_v3.py` es compartido; V4 usa su propio `qa_v4.py`. Las utilidades de umbral importan `medir` de `versiones.v1.evaluar`.

## Comandos desde la raíz (PowerShell)

```powershell
.\.venv\Scripts\python.exe versiones/v3/entrenar_v3.py
.\.venv\Scripts\python.exe versiones/v3/evaluar_v3.py
```

El primer comando entrena; no se ejecutó durante esta reorganización. La evaluación escribe registros: revisa sus destinos antes de repetirla si deseas preservar una corrida. La opción `--comprobar` del entrenador no entrena, pero sí puede crear registros. 

## Resultados y decisión

Conservó V2: el candidato empeoró EM con respuesta, aunque mejoró la media armónica.

Fuente: `logs/roberta_squad2_v3/resumen.json`. Selección registrada: `parent_v2`.

Candidato en las 1000 preguntas de seguimiento:

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 1000 | 79.2000 | 81.2525 |
| answerable | 486 | 73.2510 | 77.4743 |
| unanswerable | 514 | 84.8249 | 84.8249 |

Referencia V2 en las mismas preguntas:

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 1000 | 78.6000 | 80.9360 |
| answerable | 486 | 73.6626 | 78.4691 |
| unanswerable | 514 | 83.2685 | 83.2685 |

Registro posterior: `logs/roberta_squad2_v3/metricas_evaluacion.json`. 

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 9873 | 79.0540 | 82.2519 |
| answerable | 4949 | 74.1564 | 80.5361 |
| unanswerable | 4924 | 83.9764 | 83.9764 |

Las configuraciones, historiales y gráficas disponibles se conservan en `logs/roberta_squad2_v3/`. Los umbrales históricos alternativos no deben confundirse con `logs/roberta_squad2_v2/umbral.json`, que usa la API. Las particiones de desarrollo y las 9873 preguntas fueron reutilizadas; no son un test independiente. No se inventan métricas faltantes.
