# QA extractivo con RoBERTa

La API utiliza V2 en `outputs/roberta_squad2_v2/modelo` y el umbral vigente de `logs/roberta_squad2_v2/umbral.json` (-0.5847749710083008). Reorganizar los scripts no cambia pesos, umbrales ni resultados.

## Estructura

```text
app.py, probar_api.py            API y prueba HTTP
project_paths.py                 raíz estable del proyecto
qa_v3.py                         inferencia, métricas y selección compartidas
descargar.py                     preparación de modelo base y dataset
comparar_modelos.py              diagnóstico entre versiones
probar_velocidad.py              medición con el modelo base
versiones/{v1,v2,v3,v4_lora,v5,v6}/  scripts y documentación por experimento
data/, models/, outputs/, logs/  artefactos en sus ubicaciones originales
tools/                          ejecutables locales excluidos de Git
verificaciones/                 inventario y validación de la reorganización
```

## Dependencias y uso

Se revisaron imports y dependencias: torch, transformers, datasets, accelerate, numpy, tqdm, matplotlib y safetensors; PEFT es adicional para LoRA. La API requiere fastapi, pydantic v2 y uvicorn. `probar_api.py` usa la biblioteca estándar. No se reinstalaron paquetes ni se cambió PyTorch/CUDA. Las versiones instaladas quedan en `verificaciones/dependencias.json`; no son una garantía de portabilidad ni un archivo lock.

Desde la raíz, en PowerShell:

```powershell
$env:QA_API_TOKEN = "tu-token-local"
$env:QA_TEAM = "team-XX"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

En otra terminal con el mismo token:

```powershell
.\.venv\Scripts\python.exe probar_api.py
```

Para iniciar uvicorn desde otro directorio, usa `--app-dir C:\proyecto-qa` y la ruta absoluta al Python del entorno. Los scripts admiten ruta absoluta desde cualquier directorio, o `python -m versiones.v3.evaluar_v3` desde la raíz. `descargar.py` descarga artefactos; los comandos de entrenamiento de cada README entrenan y escriben salidas. No se ejecutaron durante la reorganización.

## Comparación y selección

| Versión | Objetivo | Decisión |
|---|---|---|
| [v1](versiones/v1/README.md) | Ajuste completo inicial de FacebookAI/roberta-base en SQuAD 2.0. | Referencia inicial; la API actual utiliza V2. No hay un resumen formal de selección de V1. |
| [v2](versiones/v2/README.md) | Ajuste completo revisado del modelo base; versión usada por la API. | Seleccionada para la API. Los diagnósticos y comparaciones de umbrales no cambian automáticamente la configuración vigente. |
| [v3](versiones/v3/README.md) | Continuación conservadora desde V2. | Conservó V2: el candidato empeoró EM con respuesta, aunque mejoró la media armónica. |
| [v4_lora](versiones/v4_lora/README.md) | Adaptación LoRA sobre V2 y exportación del candidato fusionado. | Conservó V2: el candidato empeoró EM con respuesta. La comprobación de fusión registrada fue satisfactoria. |
| [v5](versiones/v5/README.md) | Prueba corta con minería de errores de TRAIN. | Seleccionó v5_candidate en seguimiento, pero no se adoptó en la API: en las 9873 preguntas subió EM global y sin respuesta, bajaron EM/F1 con respuesta, F1 global y media armónica frente a V2. |
| [v6](versiones/v6/README.md) | Refuerzo de preguntas con respuesta y protección de F1. | Conservó V2: el candidato empeoró EM sin respuesta y media armónica. No se encontró evaluación posterior de V6 en 9873 preguntas. |

V3, V4 y V6 conservaron V2: sus candidatos no son los modelos finalmente seleccionados. V5 sí fue seleccionado en seguimiento (EM 79.5 frente a 78.6 de V2), pero no sostuvo una mejora equilibrada en las 9873 preguntas. Allí V5 obtuvo EM/F1 global 79.1249/82.0755, frente a 79.0540/82.2519 de V2 conservada por V3; EM con respuesta bajó de 74.1564 a 73.4088, mientras EM sin respuesta subió de 83.9764 a 84.8700. Fuente: `logs/roberta_squad2_v5_rapido/metricas_evaluacion.json` y `logs/roberta_squad2_v3/metricas_evaluacion.json` (`source: parent_v2`). La API continúa en V2. Los JSON `metricas_umbral_final.json` documentan otros umbrales históricos: no representan necesariamente el umbral vigente.

## Archivos revisados y conservación

`versiones/v1/app_historica.py` conserva la API histórica V1 (antes `.py` en la raíz), con rutas e imports adaptados. No es la API activa; `app.py` en la raíz sigue utilizando V2. `request.json` es un cuerpo de petición; `public_sample.json` contiene ejemplos públicos; `datos_calibracion.json` contiene datos de calibración. Se conservaron todos. `roberta_v3/` se eliminó después de comprobar nuevamente que estaba vacía; los experimentos están en `versiones/`. Los cachés Python no son código fuente y se excluyen de Git. `tools/cloudflared.exe` es una herramienta de túnel y se conserva localmente, excluida. No se encontraron otros archivos temporales en la raíz. Las dos utilidades puntuales usadas para migrar y generar documentación se retiraron después de comprobar su función; solo se eliminaron esos archivos nuevos creados durante esta tarea.

Se mantienen métricas, configuraciones, historiales y gráficas de logs disponibles para Git; se excluyen predicciones voluminosas, datos, pesos, entornos, ejecutables y secretos. El historial Git y los cambios locales se preservan; no se hicieron commits ni push.

## Verificación sin entrenar

` .\.venv\Scripts\python.exe verificar_organizacion.py ` comprueba sintaxis e imports desde otro directorio y carga V2 en CPU con una inferencia de calentamiento, sin levantar un servidor. También contrasta tamaño y fecha de los 342 archivos protegidos con el inventario previo, y SHA-256 de los logs. El resultado está en `verificaciones/resultado.json`. No se ejecutan entrenamientos, evaluaciones masivas ni el generador histórico V2.
