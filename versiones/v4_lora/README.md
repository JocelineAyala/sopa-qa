# RoBERTa v4_lora

Adaptación LoRA sobre V2 y exportación del candidato fusionado.

## Archivos y cambios

Rango 8, alpha 16, dropout 0.05; query/value y cabeza QA. Learning rate LoRA 1e-5 y cabeza 1e-6.

- `entrenar_v4.py`
- `evaluar_v4.py`
- `finalizar_v4.py`
- `qa_v4.py`
- `README_V4.md` — documentación original preservada; sus instrucciones de ubicación fueron actualizadas.

Usa `project_paths.ROOT`; no depende del directorio actual. Los artefactos continúan en la raíz, sin mover datos, modelos, salidas ni logs. `qa_v3.py` es compartido; V4 usa su propio `qa_v4.py`. Las utilidades de umbral importan `medir` de `versiones.v1.evaluar`.

## Comandos desde la raíz (PowerShell)

```powershell
.\.venv\Scripts\python.exe versiones/v4_lora/entrenar_v4.py
.\.venv\Scripts\python.exe versiones/v4_lora/evaluar_v4.py
```

El primer comando entrena; no se ejecutó durante esta reorganización. La evaluación escribe registros: revisa sus destinos antes de repetirla si deseas preservar una corrida. La opción `--comprobar` del entrenador no entrena, pero sí puede crear registros. 

## Resultados y decisión

Conservó V2: el candidato empeoró EM con respuesta. La comprobación de fusión registrada fue satisfactoria.

Fuente: `logs/roberta_squad2_v4_lora/resumen.json`. Selección registrada: `parent_v2`.

Candidato en las 1000 preguntas de seguimiento:

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 1000 | 78.7000 | 80.8157 |
| answerable | 486 | 73.2510 | 77.6044 |
| unanswerable | 514 | 83.8521 | 83.8521 |

Referencia V2 en las mismas preguntas:

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 1000 | 78.6000 | 80.9360 |
| answerable | 486 | 73.6626 | 78.4691 |
| unanswerable | 514 | 83.2685 | 83.2685 |

Registro posterior: `logs/roberta_squad2_v4_lora/evaluacion_9873_candidato.json`. 

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 9873 | 79.1755 | 82.0551 |
| answerable | 4949 | 73.6714 | 79.4161 |
| unanswerable | 4924 | 84.7076 | 84.7076 |

Las configuraciones, historiales y gráficas disponibles se conservan en `logs/roberta_squad2_v4_lora/`. Los umbrales históricos alternativos no deben confundirse con `logs/roberta_squad2_v2/umbral.json`, que usa la API. Las particiones de desarrollo y las 9873 preguntas fueron reutilizadas; no son un test independiente. No se inventan métricas faltantes.
