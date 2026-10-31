# RoBERTa v6

Refuerzo de preguntas con respuesta y protección de F1.

## Archivos y cambios

Mayor exposición a errores positivos; exige además no empeorar F1 con respuesta ni global.

- `entrenar_v6.py`

Usa `project_paths.ROOT`; no depende del directorio actual. Los artefactos continúan en la raíz, sin mover datos, modelos, salidas ni logs. `qa_v3.py` es compartido; V4 usa su propio `qa_v4.py`. Las utilidades de umbral importan `medir` de `versiones.v1.evaluar`.

## Comandos desde la raíz (PowerShell)

```powershell
.\.venv\Scripts\python.exe versiones/v6/entrenar_v6.py
.\.venv\Scripts\python.exe versiones/v6/entrenar_v6.py --evaluar
```

El primer comando entrena; no se ejecutó durante esta reorganización. La evaluación escribe registros: revisa sus destinos antes de repetirla si deseas preservar una corrida. 

## Resultados y decisión

Conservó V2: el candidato empeoró EM sin respuesta y media armónica. No se encontró evaluación posterior de V6 en 9873 preguntas.

Fuente: `logs/roberta_squad2_v6/resumen.json`. Selección registrada: `parent_v2`.

Candidato en las 1000 preguntas de seguimiento:

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 1000 | 78.0000 | 80.3588 |
| answerable | 486 | 74.0741 | 78.9275 |
| unanswerable | 514 | 81.7121 | 81.7121 |

Referencia V2 en las mismas preguntas:

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 1000 | 78.6000 | 80.9360 |
| answerable | 486 | 73.6626 | 78.4691 |
| unanswerable | 514 | 83.2685 | 83.2685 |

Las configuraciones, historiales y gráficas disponibles se conservan en `logs/roberta_squad2_v6/`. Los umbrales históricos alternativos no deben confundirse con `logs/roberta_squad2_v2/umbral.json`, que usa la API. Las particiones de desarrollo y las 9873 preguntas fueron reutilizadas; no son un test independiente. No se inventan métricas faltantes.
