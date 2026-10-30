import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

#Usamos el modelo que ya descargaste.
ruta = "models/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(ruta)
model = AutoModelForQuestionAnswering.from_pretrained(ruta)
model.to("cpu")
model.eval()

#Probamos el contexto largo entregado por el docente.
with open("public_sample.json", encoding="utf-8") as archivo:
    ejemplos = json.load(archivo)["items"]

ejemplo = next(x for x in ejemplos if x["id"] == "sample-05")

def ejecutar():
    entradas = tokenizer(
        ejemplo["question"],
        ejemplo["context"],
        max_length=384,
        stride=128,
        truncation="only_second",
        return_overflowing_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    entradas.pop("overflow_to_sample_mapping")

    with torch.inference_mode():
        model(**entradas)

    return entradas["input_ids"].shape[0]

#La primera ejecución sirve para calentar el modelo.
print("Preparando prueba en CPU...")
ventanas = ejecutar()

tiempos = []
for numero in range(3):
    inicio = time.perf_counter()
    ejecutar()
    duracion = time.perf_counter() - inicio
    tiempos.append(duracion)
    print(f"Prueba {numero + 1}: {duracion:.3f} segundos")

print(f"\nVentanas procesadas: {ventanas}")
print(f"Promedio: {sum(tiempos) / len(tiempos):.3f} segundos")