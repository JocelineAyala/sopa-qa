"""API histórica V1; la API activa es app.py en la raíz y utiliza V2."""
import sys
from pathlib import Path as _ProjectPath

_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import asyncio
import json
import logging
import math
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from versiones.v1.evaluar import predecir, MAX_LENGTH, DOC_STRIDE, MAX_ANSWER_LENGTH


logger = logging.getLogger("uvicorn.error")
MODELO = ROOT / "outputs/roberta_squad2/modelo"
CONFIGURACION = ROOT / "logs/roberta_squad2/umbral.json"

estado = {"ready": False}
candado = threading.Lock()


class PredictRequest(BaseModel):
    #Rechazamos campos adicionales y tipos incorrectos.
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1, max_length=64)
    context: str = Field(min_length=1, max_length=5000)
    question: str = Field(min_length=1, max_length=500)


class PredictResponse(BaseModel):
    id: str
    answer: str | None
    start: int | None = Field(ge=0)
    score: float = Field(ge=0, le=1)


def cargar():
    with open(CONFIGURACION, encoding="utf-8") as archivo:
        configuracion = json.load(archivo)

    #La inferencia debe coincidir con la usada al calibrar.
    assert configuracion["max_seq_length"] == MAX_LENGTH
    assert configuracion["doc_stride"] == DOC_STRIDE
    assert configuracion["max_answer_length"] == MAX_ANSWER_LENGTH
    assert configuracion["top_k"] == 20

    umbral = float(configuracion["null_threshold"])

    if not math.isfinite(umbral):
        raise ValueError("El umbral debe ser finito.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODELO,
        use_fast=True,
        local_files_only=True,
    )
    modelo = AutoModelForQuestionAnswering.from_pretrained(
        MODELO,
        local_files_only=True,
    )
    dispositivo = torch.device("cpu")
    modelo.to(dispositivo)
    modelo.eval()

    #Calentamos el modelo antes de anunciar que está listo.
    predecir(
        modelo,
        tokenizer,
        dispositivo,
        "The satellite was launched in 2024.",
        "When was the satellite launched?",
    )

    return {
        "model": modelo,
        "tokenizer": tokenizer,
        "device": dispositivo,
        "threshold": umbral,
        "params": sum(p.numel() for p in modelo.parameters()),
    }


async def preparar_modelo():
    try:
        recursos = await asyncio.to_thread(cargar)
        estado.update(recursos)
        estado["ready"] = True
        logger.info("Modelo listo en CPU.")
    except Exception:
        logger.exception("No se pudo cargar el modelo.")
        raise


@asynccontextmanager
async def lifespan(app):
    token = os.environ.get("QA_API_TOKEN")

    if not token:
        raise RuntimeError("Debés definir QA_API_TOKEN antes de iniciar.")

    equipo = os.environ.get("QA_TEAM", "team-XX")

    if not equipo or any(caracter.isspace() for caracter in equipo):
        raise RuntimeError("QA_TEAM debe tener un nombre sin espacios.")

    estado["token"] = token
    estado["team"] = equipo
    estado["ready"] = False

    #Permitimos que health devuelva loading mientras se carga.
    tarea = asyncio.create_task(preparar_modelo())
    estado["load_task"] = tarea

    try:
        yield
    finally:
        #Esperamos la carga si todavía está en curso.
        await asyncio.gather(tarea, return_exceptions=True)
        estado["ready"] = False


app = FastAPI(
    title="Extractive QA Service",
    lifespan=lifespan,
)


@app.middleware("http")
async def autenticar_y_manejar_errores(request: Request, call_next):
    if request.url.path in {"/info", "/predict"}:
        cabeceras = request.headers.getlist("authorization")
        esperado = "Bearer " + estado["token"]

        autorizado = (
            len(cabeceras) == 1
            and secrets.compare_digest(
                cabeceras[0].encode("utf-8"),
                esperado.encode("utf-8"),
            )
        )

        if not autorizado:
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized"},
                headers={"WWW-Authenticate": "Bearer"},
            )

    try:
        return await call_next(request)
    except Exception:
        logger.exception("Error procesando la solicitud.")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error"},
        )


@app.exception_handler(RequestValidationError)
async def cuerpo_invalido(request: Request, error: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "bad_request",
            "detail": "Invalid JSON body, fields, types or lengths.",
        },
    )


@app.get("/health")
async def health():
    if estado["ready"]:
        return {"status": "ok"}

    return JSONResponse(
        status_code=503,
        content={"status": "loading"},
    )


async def esperar_modelo():
    #Las rutas protegidas esperan la carga si aún no termina.
    await asyncio.shield(estado["load_task"])


@app.get("/info")
async def info():
    await esperar_modelo()

    return {
        "team": estado["team"],
        "model": "FacebookAI/roberta-base",
        "params": estado["params"],
        "max_seq_length": MAX_LENGTH,
        "doc_stride": DOC_STRIDE,
    }


def resolver(peticion: PredictRequest):
    #Evitamos ejecutar varias inferencias simultáneas en CPU.
    with candado:
        texto, inicio, diferencia = predecir(
            estado["model"],
            estado["tokenizer"],
            estado["device"],
            peticion.context,
            peticion.question,
        )

    margen = diferencia - estado["threshold"]

    if margen > 0:
        texto = None
        inicio = None
    else:
        assert (
            peticion.context[inicio:inicio + len(texto)] == texto
        )

    #Score orientativo basado en la distancia al umbral.
    #No es una probabilidad calibrada y el docente no lo puntúa.
    score = 1.0 / (1.0 + math.exp(-min(abs(margen), 60.0)))

    return {
        "id": peticion.id,
        "answer": texto,
        "start": inicio,
        "score": score,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(peticion: PredictRequest):
    await esperar_modelo()
    return await asyncio.to_thread(resolver, peticion)
