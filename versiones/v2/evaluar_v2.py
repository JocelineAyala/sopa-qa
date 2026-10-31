
# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import json
import math
import re
import string
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

RUTA_MODELO = str(ROOT / "outputs/roberta_squad2_v2/modelo")
RUTA_RESULTADOS = ROOT / "logs/roberta_squad2_v2"
MAX_LENGTH = 384
DOC_STRIDE = 128
MAX_ANSWER_LENGTH = 30


def normalizar(texto):
    #Normalización usada para comparar respuestas al estilo SQuAD.
    texto = texto.lower()
    texto = "".join(
        caracter for caracter in texto
        if caracter not in string.punctuation
    )
    texto = re.sub(r"\b(a|an|the)\b", " ", texto)
    return " ".join(texto.split())


def puntuar(respuesta, referencias):
    #Aceptamos cualquiera de las respuestas de referencia.
    referencias = [
        texto for texto in referencias if normalizar(texto)
    ] or [""]

    prediccion = normalizar(respuesta).split()
    mejor_em = 0.0
    mejor_f1 = 0.0

    for referencia in referencias:
        esperado = normalizar(referencia).split()
        em = float(prediccion == esperado)

        if not prediccion or not esperado:
            f1 = float(prediccion == esperado)
        else:
            comunes = sum(
                (Counter(prediccion) & Counter(esperado)).values()
            )

            if comunes == 0:
                f1 = 0.0
            else:
                precision = comunes / len(prediccion)
                recall = comunes / len(esperado)
                f1 = 2 * precision * recall / (precision + recall)

        mejor_em = max(mejor_em, em)
        mejor_f1 = max(mejor_f1, f1)

    return mejor_em, mejor_f1


def predecir(modelo, tokenizer, dispositivo, contexto, pregunta):
    entradas = tokenizer(
        pregunta.strip(),
        contexto,
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        truncation="only_second",
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )

    offsets = entradas.pop("offset_mapping").tolist()
    entradas.pop("overflow_to_sample_mapping")

    secuencias = [
        entradas.sequence_ids(i) for i in range(len(offsets))
    ]

    ids = entradas["input_ids"].tolist()
    tensores = {
        clave: valor.to(dispositivo)
        for clave, valor in entradas.items()
    }

    with torch.inference_mode():
        salida = modelo(**tensores)

    inicios = salida.start_logits.float().cpu()
    finales = salida.end_logits.float().cpu()

    mejor_puntuacion = float("-inf")
    menor_null = float("inf")
    mejor_texto = None
    mejor_inicio = None

    for i, mapa in enumerate(offsets):
        cls = ids[i].index(tokenizer.cls_token_id)
        puntuacion_null = float(inicios[i, cls] + finales[i, cls])
        menor_null = min(menor_null, puntuacion_null)

        validos = [
            secuencia == 1 and mapa[j][1] > mapa[j][0]
            for j, secuencia in enumerate(secuencias[i])
        ]
        mascara = torch.tensor(validos, dtype=torch.bool)

        inicio_logits = inicios[i].masked_fill(
            ~mascara, float("-inf")
        )
        final_logits = finales[i].masked_fill(
            ~mascara, float("-inf")
        )

        cantidad = min(20, len(mapa))
        candidatos_inicio = inicio_logits.topk(cantidad).indices.tolist()
        candidatos_final = final_logits.topk(cantidad).indices.tolist()

        for inicio in candidatos_inicio:
            for final in candidatos_final:
                if not validos[inicio] or not validos[final]:
                    continue
                if final < inicio:
                    continue
                if final - inicio + 1 > MAX_ANSWER_LENGTH:
                    continue

                desde = mapa[inicio][0]
                hasta = mapa[final][1]

                if hasta <= desde:
                    continue

                puntuacion = float(
                    inicio_logits[inicio] + final_logits[final]
                )

                if puntuacion > mejor_puntuacion:
                    mejor_puntuacion = puntuacion
                    mejor_texto = contexto[desde:hasta]
                    mejor_inicio = desde

    if mejor_texto is None:
        raise RuntimeError("No se encontró ningún fragmento candidato.")

    #Un valor mayor favorece responder null.
    diferencia = menor_null - mejor_puntuacion

    if not math.isfinite(diferencia):
        raise RuntimeError("Se obtuvo una puntuación no finita.")

    assert (
        contexto[mejor_inicio:mejor_inicio + len(mejor_texto)]
        == mejor_texto
    )

    return mejor_texto, mejor_inicio, diferencia


def obtener_predicciones(datos, modelo, tokenizer, dispositivo, nombre):
    filas = []

    for ejemplo in tqdm(datos, desc=nombre):
        texto, inicio, diferencia = predecir(
            modelo,
            tokenizer,
            dispositivo,
            ejemplo["context"],
            ejemplo["question"],
        )

        referencias = ejemplo["answers"]["text"]
        em_span, f1_span = puntuar(texto, referencias)
        em_null, f1_null = puntuar("", referencias)

        filas.append({
            "id": ejemplo["id"],
            "span": texto,
            "start": inicio,
            "difference": diferencia,
            "has_answer": bool(referencias),
            "references": referencias,
            "em_span": em_span,
            "f1_span": f1_span,
            "em_null": em_null,
            "f1_null": f1_null,
        })

    return filas


def elegir_umbral(filas):
    #Probamos todos los cambios de decisión posibles en calibración.
    ordenadas = sorted(filas, key=lambda fila: fila["difference"])
    total_em = sum(fila["em_null"] for fila in filas)
    total_f1 = sum(fila["f1_null"] for fila in filas)

    umbral = math.nextafter(
        ordenadas[0]["difference"], float("-inf")
    )
    mejor_umbral = umbral
    mejor_resultado = (total_f1, total_em)

    i = 0

    while i < len(ordenadas):
        diferencia = ordenadas[i]["difference"]

        #Con diferencia <= umbral elegimos el fragmento.
        while (
            i < len(ordenadas)
            and ordenadas[i]["difference"] == diferencia
        ):
            fila = ordenadas[i]
            total_em += fila["em_span"] - fila["em_null"]
            total_f1 += fila["f1_span"] - fila["f1_null"]
            i += 1

        resultado = (total_f1, total_em)

        if resultado > mejor_resultado:
            mejor_resultado = resultado
            mejor_umbral = diferencia

    return mejor_umbral


def medir(filas, umbral):
    def resumen(grupo):
        if not grupo:
            return {"count": 0, "EM": None, "F1": None}

        total_em = 0.0
        total_f1 = 0.0
        cantidad_null = 0

        for fila in grupo:
            sin_respuesta = fila["difference"] > umbral
            sufijo = "null" if sin_respuesta else "span"

            total_em += fila[f"em_{sufijo}"]
            total_f1 += fila[f"f1_{sufijo}"]
            cantidad_null += int(sin_respuesta)

        return {
            "count": len(grupo),
            "EM": 100 * total_em / len(grupo),
            "F1": 100 * total_f1 / len(grupo),
            "predicted_null": cantidad_null,
        }

    return {
        "overall": resumen(filas),
        "answerable": resumen([
            fila for fila in filas if fila["has_answer"]
        ]),
        "unanswerable": resumen([
            fila for fila in filas if not fila["has_answer"]
        ]),
    }


def guardar(ruta, contenido):
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(
            contenido,
            archivo,
            ensure_ascii=False,
            indent=2,
        )


def main():
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    dispositivo = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Dispositivo de evaluación:", dispositivo)

    tokenizer = AutoTokenizer.from_pretrained(
        RUTA_MODELO, use_fast=True
    )
    modelo = AutoModelForQuestionAnswering.from_pretrained(
        RUTA_MODELO
    )
    modelo.to(dispositivo)
    modelo.eval()

    #Reproducimos el orden usado para monitorear el entrenamiento.
    validacion = (
        load_from_disk(str(ROOT / "data/squad_v2"))["validation"]
        .shuffle(seed=42)
    )

    calibracion = validacion.select(range(2000))
    evaluacion = validacion.select(range(2000, len(validacion)))

    print("Preguntas para calibrar:", len(calibracion))
    print("Preguntas para evaluar:", len(evaluacion))

    filas_calibracion = obtener_predicciones(
        calibracion, modelo, tokenizer, dispositivo, "Calibración"
    )
    guardar(
        RUTA_RESULTADOS / "predicciones_calibracion.json",
        filas_calibracion,
    )

    umbral = elegir_umbral(filas_calibracion)

    configuracion = {
        "model_path": RUTA_MODELO,
        "null_threshold": umbral,
        "null_rule": "return null if difference > null_threshold",
        "difference": "minimum CLS score minus best span score",
        "max_seq_length": MAX_LENGTH,
        "doc_stride": DOC_STRIDE,
        "max_answer_length": MAX_ANSWER_LENGTH,
        "top_k": 20,
        "selection_metric": "calibration F1, then EM",
        "calibration_count": len(calibracion),
        "evaluation_count": len(evaluacion),
    }

    guardar(
        RUTA_RESULTADOS / "umbral.json",
        configuracion,
    )

    print(f"\nUmbral elegido: {umbral:.4f}")

    filas_evaluacion = obtener_predicciones(
        evaluacion, modelo, tokenizer, dispositivo, "Evaluación"
    )
    guardar(
        RUTA_RESULTADOS / "predicciones_evaluacion.json",
        filas_evaluacion,
    )

    resultados = {
        "configuration": configuracion,
        "calibration_threshold_0": medir(filas_calibracion, 0.0),
        "calibration_tuned": medir(filas_calibracion, umbral),
        "evaluation_threshold_0": medir(filas_evaluacion, 0.0),
        "evaluation_tuned": medir(filas_evaluacion, umbral),
    }

    guardar(
        RUTA_RESULTADOS / "metricas.json",
        resultados,
    )

    print("\nRESULTADOS EN LAS 9873 PREGUNTAS DE EVALUACIÓN")

    for nombre, clave in [
        ("Umbral inicial", "evaluation_threshold_0"),
        ("Umbral calibrado", "evaluation_tuned"),
    ]:
        print(f"\n{nombre}:")

        for grupo, valores in resultados[clave].items():
            print(
                f"  {grupo}: n={valores['count']} | "
                f"EM={valores['EM']:.2f} | "
                f"F1={valores['F1']:.2f}"
            )

    print("\nResultados guardados en:", RUTA_RESULTADOS)


if __name__ == "__main__":
    main()