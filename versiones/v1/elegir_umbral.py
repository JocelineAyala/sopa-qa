
# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import json
import math
from pathlib import Path

from versiones.v1.evaluar import medir

CARPETA = ROOT / "logs/roberta_squad2"


def leer(nombre):
    with open(CARPETA / nombre, encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar(nombre, contenido):
    with open(
        CARPETA / nombre,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(contenido, archivo, indent=2, ensure_ascii=False)


def ponderar(metricas, metrica):
    #Representamos la proporción estimada por el docente.
    return (
        0.75 * metricas["answerable"][metrica]
        + 0.25 * metricas["unanswerable"][metrica]
    )


def main():
    calibracion = leer("predicciones_calibracion.json")
    evaluacion = leer("predicciones_evaluacion.json")

    diferencias = sorted({
        fila["difference"] for fila in calibracion
    })

    candidatos = [
        math.nextafter(diferencias[0], float("-inf")),
        *diferencias,
    ]

    mejor_umbral = None
    mejor_puntaje = None
    mejores_metricas = None

    print("Buscando el umbral únicamente con calibración...")

    for umbral in candidatos:
        metricas = medir(calibracion, umbral)

        #Priorizamos EM; F1 desempata.
        puntaje = (
            round(ponderar(metricas, "EM"), 10),
            round(ponderar(metricas, "F1"), 10),
        )

        if mejor_puntaje is None or puntaje > mejor_puntaje:
            mejor_puntaje = puntaje
            mejor_umbral = umbral
            mejores_metricas = metricas

    #El umbral queda fijado antes de evaluar las otras preguntas.
    configuracion = leer("umbral.json")
    configuracion["null_threshold"] = mejor_umbral
    configuracion["selection_metric"] = (
        "Calibration weighted EM; weighted F1 breaks ties"
    )
    configuracion["expected_answerable_fraction"] = 0.75
    configuracion["expected_unanswerable_fraction"] = 0.25

    guardar("umbral.json", configuracion)

    metricas_evaluacion = medir(evaluacion, mejor_umbral)

    resultados = {
        "threshold": mejor_umbral,
        "calibration": mejores_metricas,
        "evaluation": metricas_evaluacion,
        "evaluation_weighted_EM": ponderar(metricas_evaluacion, "EM"),
        "evaluation_weighted_F1": ponderar(metricas_evaluacion, "F1"),
    }

    guardar("metricas_umbral_final.json", resultados)

    print(f"\nUmbral seleccionado: {mejor_umbral:.4f}")
    print("\nEVALUACIÓN CON EL UMBRAL FIJADO:")

    for grupo, valores in metricas_evaluacion.items():
        print(
            f"{grupo}: n={valores['count']} | "
            f"EM={valores['EM']:.2f} | "
            f"F1={valores['F1']:.2f}"
        )

    print(
        "\nEM ponderado para una mezcla 75/25:",
        f"{resultados['evaluation_weighted_EM']:.2f}",
    )
    print(
        "F1 ponderado para una mezcla 75/25:",
        f"{resultados['evaluation_weighted_F1']:.2f}",
    )
    print("\nConfiguración guardada en:", CARPETA / "umbral.json")


if __name__ == "__main__":
    main()