# RoBERTa v1

Ajuste completo inicial de FacebookAI/roberta-base en SQuAD 2.0.

## Archivos y cambios

3 épocas, learning rate 2e-5, acumulación 4 y warmup 4944 pasos.

- `elegir_umbral.py`
- `entrenar.py`
- `evaluar.py`
- `probar_entrenamiento.py`
- `probar_modelo.py`
- `app_historica.py`: conserva la API V1, antes llamada `.py` en la raíz. Usa `outputs/roberta_squad2/modelo` y `logs/roberta_squad2/umbral.json`. No es la API activa: el servicio actual es `app.py` en la raíz y utiliza V2.

Usa `project_paths.ROOT`; no depende del directorio actual. Los artefactos continúan en la raíz, sin mover datos, modelos, salidas ni logs. `qa_v3.py` es compartido; V4 usa su propio `qa_v4.py`. Las utilidades de umbral importan `medir` de `versiones.v1.evaluar`.

## Comandos desde la raíz (PowerShell)

```powershell
.\.venv\Scripts\python.exe versiones/v1/entrenar.py
.\.venv\Scripts\python.exe versiones/v1/evaluar.py
```

El primer comando entrena; no se ejecutó durante esta reorganización. La evaluación escribe registros: revisa sus destinos antes de repetirla si deseas preservar una corrida. 

## Resultados y decisión

Referencia inicial; la API actual utiliza V2. No hay un resumen formal de selección de V1.

Registro posterior: `logs/roberta_squad2/metricas.json`. 

| Grupo | Preguntas | EM (%) | F1 (%) |
|---|---:|---:|---:|
| overall | 9873 | 77.0688 | 80.5217 |
| answerable | 4949 | 76.2174 | 83.1059 |
| unanswerable | 4924 | 77.9245 | 77.9245 |

Las configuraciones, historiales y gráficas disponibles se conservan en `logs/roberta_squad2/`. Los umbrales históricos alternativos no deben confundirse con `logs/roberta_squad2_v2/umbral.json`, que usa la API. Las particiones de desarrollo y las 9873 preguntas fueron reutilizadas; no son un test independiente. No se inventan métricas faltantes.
