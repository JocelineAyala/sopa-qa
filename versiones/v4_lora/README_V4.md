# RoBERTa V4 con LoRA

Experimento sobre nuestra V2 para QA extractivo SQuAD 2.0. No descarga un modelo ya ajustado en SQuAD ni cambia la API. No garantiza superar V2.

## Ubicacion

Los archivos de este experimento están en `versiones/v4_lora/`:

- `entrenar_v4.py`
- `qa_v4.py`
- `evaluar_v4.py`
- `README_V4.md`

Las rutas usan la raíz del proyecto y admiten ejecución desde cualquier directorio mediante una ruta absoluta al script.

Se necesita la V2 original en `outputs/roberta_squad2_v2/modelo`, su umbral en `logs/roberta_squad2_v2/umbral.json` y el dataset en `data/squad_v2`.

## Dependencia adicional

En PowerShell, dentro de `C:\proyecto-qa`:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps peft
```

Este comando instala PEFT sin cambiar PyTorch, Transformers ni sus otras dependencias. El proyecto ya usa torch, transformers, datasets, accelerate, safetensors y matplotlib. Si aparece un error de importacion o version, comparte el mensaje: no actualices todas las librerias ni reinstales PyTorch a ciegas. Se registran las versiones realmente usadas.

## Comprobar primero

```powershell
.\.venv\Scripts\python.exe versiones/v4_lora/entrenar_v4.py --comprobar
```

Valida CUDA, la carga de V2, la equivalencia inicial al insertar LoRA, que solo las matrices LoRA y la cabeza QA sean entrenables, y que ambas reciban gradientes finitos. Hace una pasada hacia atras sin ejecutar un paso del optimizador: no actualiza pesos. Tambien prepara datos y comprueba las metricas con Trainer. Debe terminar con `COMPROBACION CORRECTA`.

La comprobacion crea registros V4, pero no exporta un modelo candidato.

## Entrenar

Si la comprobacion termina correctamente:

```powershell
.\.venv\Scripts\python.exe versiones/v4_lora/entrenar_v4.py
```

- Base: nuestros pesos V2, procedentes originalmente de `FacebookAI/roberta-base`.
- LoRA: rango 8, alpha 16, dropout 0.05, aplicado a `query` y `value`.
- Cabeza `qa_outputs` entrenable y guardada con el adaptador.
- Learning rate LoRA: `1e-5`; cabeza QA: `1e-6`.
- Batch fisico 2, acumulacion 8: batch efectivo 16 en una GPU.
- Una epoca como maximo, warmup 5 %, AdamW y calendario lineal.
- FP16 en entrenamiento, gradient checkpointing no reentrante.
- Evaluacion y guardado cada 500 pasos.
- Parada temprana tras cuatro evaluaciones sin mejora de la media armonica de EM por clase.

No lances dos entrenamientos a la vez. Si interrumpes, el mismo comando reanuda desde el ultimo checkpoint encontrado; lo posterior a ese checkpoint se pierde. Si ya existe `resumen.json`, la corrida se considera terminada y no se repite automaticamente. La comprobacion no valida el estado del optimizador de un checkpoint interrumpido.

## Evaluacion y seleccion

Se conservan exactamente las particiones V3:

1. Primeras 1000 preguntas de `validation.shuffle(seed=42)`: elegir umbral por media armonica de EM entre clases.
2. Siguientes 1000: seleccionar checkpoint y comparar con V2.

No se entrenan pesos con estas preguntas. Como ya se usaron en otros experimentos, son desarrollo reutilizado, no un test independiente.

Trainer elige el mejor checkpoint por la media armonica. Despues se guarda el adaptador y se fusiona LoRA con V2 para exportar un modelo RoBERTa convencional. Se comprueba que la fusion preserve los logits en una entrada de prueba. El modelo fusionado se evalua sin autocast, se calibra y se compara con V2.

Se promueve el candidato solo si mejora la media armonica y no empeora EM en ninguna de las dos clases. En caso contrario, `modelo` contiene una copia de V2. Se conserva el candidato para comparar, aunque no se promueva. No se cambia la API automaticamente.

## Archivos

En `outputs/roberta_squad2_v4_lora`:

- `checkpoint-*`: checkpoints LoRA para reanudar; dependen de los pesos V2 originales.
- `adaptador_candidato`: adaptador y cabeza QA del mejor checkpoint; requiere V2.
- `candidato`: RoBERTa con LoRA fusionado; se carga sin PEFT.
- `modelo`: candidato promovido o copia de V2.

En `logs/roberta_squad2_v4_lora`:

- `configuracion.json`, `particiones.json`, `baseline_v2.json`.
- `calidad_por_checkpoint.jsonl`, `historial.json`, `loss.png`, `calidad.png`.
- `umbral_candidato.json`, `umbral.json`, `resumen.json`.
- `predicciones_desarrollo_candidato.json`.

No borres los pesos V2: son la base del adaptador. Excluye `outputs/` del repositorio.

## Comparacion posterior

Primero revisa el resumen de entrenamiento. Para comparar despues en las otras 9873 preguntas con umbral fijo:

```powershell
.\.venv\Scripts\python.exe versiones/v4_lora/evaluar_v4.py --candidato
```

Para evaluar el modelo seleccionado:

```powershell
.\.venv\Scripts\python.exe versiones/v4_lora/evaluar_v4.py
```

Estas preguntas ya fueron evaluadas antes; el test oculto del docente sigue siendo externo. No se reajusta el umbral en esta comparacion.

## Validacion del paquete

Se revisaron la sintaxis, los limites de respuesta al crear etiquetas de ventanas y el postprocesamiento QA con casos sinteticos. Este entorno de preparacion no contiene PyTorch, Transformers, PEFT ni los pesos de la laptop: no se realizo entrenamiento real ni una prueba completa de compatibilidad aqui. `--comprobar` verifica esa integracion en tu entorno antes de autorizar pasos del optimizador.

Documentacion de referencia: https://huggingface.co/docs/peft/package_reference/lora y https://huggingface.co/docs/peft/package_reference/peft_types.
