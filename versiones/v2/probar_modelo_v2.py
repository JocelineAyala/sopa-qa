
# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

RUTA = str(ROOT / "outputs/roberta_squad2_v2/modelo")

#Este umbral es provisional. Después lo calibraremos con SQuAD.
UMBRAL_NULL = json.loads(ROOT / 'logs/roberta_squad2_v2/umbral.json'.read_text(encoding='utf-8'))['null_threshold']

tokenizer = AutoTokenizer.from_pretrained(RUTA, use_fast=True)
model = AutoModelForQuestionAnswering.from_pretrained(RUTA)
model.to("cpu")
model.eval()


def responder(contexto, pregunta):
    entradas = tokenizer(
        pregunta.strip(),
        contexto,
        max_length=384,
        stride=128,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )

    offsets = entradas.pop("offset_mapping").tolist()
    entradas.pop("overflow_to_sample_mapping")

    secuencias = [
        entradas.sequence_ids(i)
        for i in range(len(offsets))
    ]

    with torch.inference_mode():
        resultado = model(**entradas)

    mejor_span = float("-inf")
    menor_null = float("inf")
    respuesta = None
    inicio_respuesta = None

    for i, mapa in enumerate(offsets):
        inicios = resultado.start_logits[i].float()
        finales = resultado.end_logits[i].float()

        ids = entradas["input_ids"][i].tolist()
        cls = ids.index(tokenizer.cls_token_id)

        #CLS representa la opción de no responder.
        puntuacion_null = float(inicios[cls] + finales[cls])
        menor_null = min(menor_null, puntuacion_null)

        #Solo permitimos respuestas ubicadas en el contexto.
        validos = torch.tensor([
            secuencia == 1 and mapa[j][1] > mapa[j][0]
            for j, secuencia in enumerate(secuencias[i])
        ])

        inicios = inicios.masked_fill(~validos, float("-inf"))
        finales = finales.masked_fill(~validos, float("-inf"))

        candidatos_inicio = torch.topk(
            inicios, k=min(20, len(inicios))
        ).indices.tolist()

        candidatos_final = torch.topk(
            finales, k=min(20, len(finales))
        ).indices.tolist()

        for inicio in candidatos_inicio:
            for final in candidatos_final:
                if not validos[inicio] or not validos[final]:
                    continue

                if final < inicio or final - inicio + 1 > 30:
                    continue

                puntuacion = float(inicios[inicio] + finales[final])
                desde = mapa[inicio][0]
                hasta = mapa[final][1]

                if hasta <= desde:
                    continue

                if puntuacion > mejor_span:
                    mejor_span = puntuacion
                    respuesta = contexto[desde:hasta]
                    inicio_respuesta = desde

    diferencia = None

    if respuesta is not None:
        diferencia = menor_null - mejor_span

        if diferencia > UMBRAL_NULL:
            respuesta = None
            inicio_respuesta = None

    return {
        "answer": respuesta,
        "start": inicio_respuesta,
        "null_score_difference": diferencia,
    }


with open(str(ROOT / "public_sample.json"), encoding="utf-8") as archivo:
    ejemplos = json.load(archivo)["items"]

resultados = []
aciertos = 0

for ejemplo in ejemplos:
    inicio = time.perf_counter()
    prediccion = responder(ejemplo["context"], ejemplo["question"])
    segundos = time.perf_counter() - inicio

    respuesta = prediccion["answer"]
    posicion = prediccion["start"]

    if ejemplo["is_impossible"]:
        coincide = respuesta is None
        esperado = None
    else:
        esperado = [x["text"] for x in ejemplo["answers"]]
        coincide = respuesta in esperado

    if respuesta is None:
        posicion_valida = posicion is None
    else:
        posicion_valida = (
            posicion is not None
            and ejemplo["context"][
                posicion:posicion + len(respuesta)
            ] == respuesta
        )

    correcto = coincide and posicion_valida
    aciertos += int(correcto)

    print("\n" + "=" * 50)
    print("ID:", ejemplo["id"])
    print("Pregunta:", ejemplo["question"])
    print("Esperado:", esperado)
    print("Respuesta:", repr(respuesta))
    print("Inicio:", posicion)
    print("Posición válida:", posicion_valida)
    print("Coincide con referencia:", correcto)
    print(f"Tiempo: {segundos:.3f} segundos")

    resultados.append({
        "id": ejemplo["id"],
        **prediccion,
        "expected": esperado,
        "offset_valid": posicion_valida,
        "reference_match": correcto,
        "seconds": segundos,
    })

carpeta = ROOT / "logs/roberta_squad2_v2"
carpeta.mkdir(parents=True, exist_ok=True)

with open(
    carpeta / "public_sample_inicial.json",
    "w",
    encoding="utf-8",
) as archivo:
    json.dump(resultados, archivo, ensure_ascii=False, indent=2)

print(f"\nCoincidencias con las referencias: {aciertos}/{len(ejemplos)}")
print("Resultados guardados en:", carpeta / "public_sample_inicial.json")