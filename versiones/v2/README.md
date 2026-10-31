# RoBERTa v2

Ajuste completo revisado del modelo base; versión usada por la API.

## Archivos y cambios

2 épocas, learning rate 1e-5, acumulación 8, warmup 824 y parada temprana con paciencia 6.

- `analizar_errores_v2.py`
- `comparar_umbrales_v2.py`
- `elegir_umbral_v2.py`
- `entrenar_v2.py`
- `evaluar_v2.py`
- `preparar_v2.py`
- `probar_modelo_v2.py`

Usa `project_paths.ROOT`; no depende del directorio actual. Los artefactos continúan en la raíz, sin mover datos, modelos, salidas ni logs. `qa_v3.py` es compartido; V4 usa su propio `qa_v4.py`. Las utilidades de umbral importan `medir` de `versiones.v1.evaluar`.

## Comandos desde la raíz (PowerShell)

```powershell
.\.venv\Scripts\python.exe versiones/v2/entrenar_v2.py
.\.venv\Scripts\python.exe versiones/v2/evaluar_v2.py
```

El primer comando entrena; no se ejecutó durante esta reorganización. La evaluación escribe registros: revisa sus destinos antes de repetirla si deseas preservar una corrida. `preparar_v2.py` es un generador histórico de V2 desde V1; se conserva y rechaza sobrescribir el script V2 existente. Los diagnósticos V2 escriben históricamente bajo logs de V3; se preservó ese destino. 

## Resultados y decisión

Seleccionada para la API. Los diagnósticos y comparaciones de umbrales no cambian automáticamente la configuración vigente.

Fuente: `logs/roberta_squad2_v2/resumen.json`. Selección registrada: `campo selected ausente`.

Registro posterior: `logs/roberta_squad2_v2/metricas.json`. 

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 9873 | 79.0540 | 82.2519 |
| answerable | 4949 | 74.1564 | 80.5361 |
| unanswerable | 4924 | 83.9764 | 83.9764 |

Las configuraciones, historiales y gráficas disponibles se conservan en `logs/roberta_squad2_v2/`. Los umbrales históricos alternativos no deben confundirse con `logs/roberta_squad2_v2/umbral.json`, que usa la API. Las particiones de desarrollo y las 9873 preguntas fueron reutilizadas; no son un test independiente. No se inventan métricas faltantes.
