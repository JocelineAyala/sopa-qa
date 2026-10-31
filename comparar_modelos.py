
from project_paths import ROOT

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from versiones.v1.evaluar import predecir, puntuar


#Pruebas nuevas de diagnóstico: no se usan para entrenar ni calibrar.
contexto_1 = (
    "ESA launched the Aurora telescope in 2018. "
    "NASA launched the Vega telescope in 2021."
)

contexto_2 = (
    "In 2020, the team attempted to launch the Orion satellite, "
    "but the launch failed. The satellite was successfully launched "
    "in 2023 from the North Station."
)

contexto_3 = (
    "The 2010 census recorded 12,000 residents in River City. "
    "The city opened its first hospital in 2015. "
    "The hospital employed 240 nurses in 2022."
)

casos = [
    (
        contexto_1,
        "Who launched the Aurora telescope?",
        ["ESA"],
    ),
    (
        contexto_1,
        "When did NASA launch the Vega telescope?",
        ["2021"],
    ),
    (
        contexto_1,
        "When did NASA launch the Aurora telescope?",
        [],
    ),
    (
        contexto_1,
        "Who manufactured the Vega telescope?",
        [],
    ),
    (
        contexto_2,
        "When was the Orion satellite successfully launched?",
        ["2023"],
    ),
    (
        contexto_2,
        "Where was the Orion satellite successfully launched from?",
        ["North Station", "the North Station"],
    ),
    (
        contexto_2,
        "From which station did the failed 2020 launch take place?",
        [],
    ),
    (
        contexto_2,
        "How much did the successful launch cost?",
        [],
    ),
    (
        contexto_3,
        "When did River City open its first hospital?",
        ["2015"],
    ),
    (
        contexto_3,
        "How many nurses did the hospital employ in 2022?",
        ["240"],
    ),
    (
        contexto_3,
        "How many doctors did the hospital employ in 2022?",
        [],
    ),
    (
        contexto_3,
        "How many residents lived in River City in 2022?",
        [],
    ),
]

versiones = [
    ("v1", "roberta_squad2"),
    ("v2", "roberta_squad2_v2"),
]

resultados = {}

for nombre, carpeta in versiones:
    ruta_modelo = Path(str(ROOT / "outputs")) / carpeta / "modelo"
    ruta_config = Path(str(ROOT / "logs")) / carpeta / "umbral.json"

    with open(ruta_config, encoding="utf-8") as archivo:
        umbral = json.load(archivo)["null_threshold"]

    tokenizer = AutoTokenizer.from_pretrained(
        ruta_modelo, use_fast=True
    )
    modelo = AutoModelForQuestionAnswering.from_pretrained(
        ruta_modelo
    )
    modelo.to("cpu")
    modelo.eval()

    print(f"\nVERSION {nombre} | Umbral: {umbral:.4f}")

    filas = []
    aciertos_con_respuesta = 0
    aciertos_sin_respuesta = 0

    for numero, (contexto, pregunta, referencias) in enumerate(casos, 1):
        texto, inicio, diferencia = predecir(
            modelo,
            tokenizer,
            torch.device("cpu"),
            contexto,
            pregunta,
        )

        if diferencia > umbral:
            texto = None
            inicio = None

        em, f1 = puntuar(
            "" if texto is None else texto,
            referencias,
        )
        correcto = em == 1.0

        if referencias:
            aciertos_con_respuesta += int(correcto)
        else:
            aciertos_sin_respuesta += int(correcto)

        print(f"\n{numero}. {pregunta}")
        print("Esperado:", referencias or None)
        print("Respuesta:", repr(texto))
        print("Coincide:", correcto)

        filas.append({
            "number": numero,
            "context": contexto,
            "question": pregunta,
            "references": referencias,
            "answer": texto,
            "start": inicio,
            "difference": diferencia,
            "EM": em,
            "F1": f1,
        })

    resultados[nombre] = {
        "threshold": umbral,
        "answerable_correct": aciertos_con_respuesta,
        "unanswerable_correct": aciertos_sin_respuesta,
        "cases": filas,
    }

    print(f"\n{nombre}: con respuesta: {aciertos_con_respuesta}/6")
    print(f"{nombre}: sin respuesta: {aciertos_sin_respuesta}/6")

    del modelo

ruta = Path(str(ROOT / "logs/comparacion_diagnostica.json"))
ruta.parent.mkdir(parents=True, exist_ok=True)
ruta.write_text(
    json.dumps(resultados, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nRESUMEN")
for nombre, datos in resultados.items():
    print(
        f"{nombre}: con respuesta "
        f"{datos['answerable_correct']}/6 | sin respuesta "
        f"{datos['unanswerable_correct']}/6"
    )

print("Guardado en:", ruta)