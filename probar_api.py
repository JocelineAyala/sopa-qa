
from project_paths import ROOT

import json
import math
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = (
    sys.argv[1] if len(sys.argv) > 1
    else "http://127.0.0.1:8000"
).rstrip("/")

TOKEN = os.environ.get("QA_API_TOKEN")

if not TOKEN:
    raise RuntimeError(
        "QA_API_TOKEN no está definida en esta terminal."
    )


def consultar(ruta, cuerpo=None, token=None):
    headers = {}

    if token is not None:
        headers["Authorization"] = "Bearer " + token

    datos = None

    if cuerpo is not None:
        datos = json.dumps(cuerpo).encode("utf-8")
        headers["Content-Type"] = "application/json"

    solicitud = Request(
        BASE_URL + ruta,
        data=datos,
        headers=headers,
        method="POST" if cuerpo is not None else "GET",
    )

    inicio = time.perf_counter()

    try:
        with urlopen(solicitud, timeout=5) as respuesta:
            estado = respuesta.status
            contenido = json.loads(
                respuesta.read().decode("utf-8")
            )
    except HTTPError as error:
        estado = error.code
        contenido = json.loads(
            error.read().decode("utf-8")
        )

    return estado, contenido, time.perf_counter() - inicio


print("PRUEBAS DEL SERVICIO")

pruebas = [
    ("/health", None, 200),
    ("/info", None, 401),
    ("/info", TOKEN + "-incorrecto", 401),
    ("/info", TOKEN, 200),
]

for ruta, token, esperado in pruebas:
    estado, contenido, segundos = consultar(ruta, token=token)

    assert estado == esperado, (
        f"{ruta}: se esperaba {esperado}, llegó {estado}"
    )

    if estado == 401:
        assert contenido.get("error") == "unauthorized"
    elif ruta == "/health":
        assert contenido.get("status") == "ok"
    else:
        assert contenido.get("team") == "sopa"
        assert contenido.get("model") == "FacebookAI/roberta-base"
        assert type(contenido.get("params")) is int
        assert contenido["params"] > 0

    print(f"{ruta}: HTTP {estado} | OK")


with open(str(ROOT / "public_sample.json"), encoding="utf-8") as archivo:
    ejemplos = json.load(archivo)["items"]

resultados = []

for ejemplo in ejemplos:
    cuerpo = {
        campo: ejemplo[campo]
        for campo in ("id", "context", "question")
    }

    try:
        estado, respuesta, segundos = consultar(
            "/predict",
            cuerpo=cuerpo,
            token=TOKEN,
        )

        assert estado == 200, f"HTTP inesperado: {estado}"
        assert {"id", "answer", "start", "score"} <= respuesta.keys()
        assert respuesta["id"] == ejemplo["id"]

        texto = respuesta["answer"]
        inicio = respuesta["start"]
        score = respuesta["score"]

        assert type(score) in (int, float)
        assert math.isfinite(score) and 0 <= score <= 1

        if texto is None:
            assert inicio is None
        else:
            assert isinstance(texto, str)
            assert type(inicio) is int and inicio >= 0
            assert (
                ejemplo["context"][inicio:inicio + len(texto)]
                == texto
            )

        referencias = [
            referencia["text"] for referencia in ejemplo["answers"]
        ]

        coincide = (
            texto is None if ejemplo["is_impossible"]
            else texto in referencias
        )

        print(
            f"\n{ejemplo['id']}: {texto!r}"
            f"\n  Contrato: OK"
            f"\n  Coincide con referencia: {coincide}"
            f"\n  Tiempo: {segundos:.3f} segundos"
        )

        resultados.append({
            "id": ejemplo["id"],
            "response": respuesta,
            "contract_valid": True,
            "reference_match": coincide,
            "seconds": segundos,
        })

    except Exception as error:
        print(f"\n{ejemplo['id']}: ERROR: {error}")

        resultados.append({
            "id": ejemplo["id"],
            "contract_valid": False,
            "reference_match": False,
            "error": str(error),
        })


#Comprobamos que una solicitud incompleta sea rechazada.
estado, contenido, _ = consultar(
    "/predict",
    cuerpo={"id": "invalida", "context": "Example text."},
    token=TOKEN,
)
assert estado in (400, 422)
assert contenido.get("error") == "bad_request"

print("\nSolicitud incompleta: rechazada correctamente.")

contrato = sum(x["contract_valid"] for x in resultados)
aciertos = sum(x["reference_match"] for x in resultados)

print(f"\nRespuestas válidas según contrato: {contrato}/5")
print(f"Coincidencias con las referencias: {aciertos}/5")

carpeta = Path(str(ROOT / "logs/roberta_squad2"))
carpeta.mkdir(parents=True, exist_ok=True)

nombre = (
    "api_local.json"
    if BASE_URL == "http://127.0.0.1:8000"
    else "api_tunel.json"
)

with open(carpeta / nombre, "w", encoding="utf-8") as archivo:
    json.dump(
        {"base_url": BASE_URL, "results": resultados},
        archivo,
        ensure_ascii=False,
        indent=2,
    )

print("Resultados guardados en:", carpeta / nombre)