
# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    set_seed(42)
    salida = ROOT / "outputs/prueba"
    salida.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "models/roberta-base"),
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        str(ROOT / "models/roberta-base")
    )

    #Tomamos una muestra pequeña del conjunto de entrenamiento.
    datos = load_from_disk(str(ROOT / "data/squad_v2"))
    muestra = datos["train"].shuffle(seed=42).select(range(256))

    def preparar(ejemplos):
        entradas = tokenizer(
            [pregunta.strip() for pregunta in ejemplos["question"]],
            ejemplos["context"],
            max_length=384,
            stride=128,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        originales = entradas.pop("overflow_to_sample_mapping")
        offsets = entradas.pop("offset_mapping")
        inicios = []
        finales = []

        for i, mapa in enumerate(offsets):
            ids = entradas["input_ids"][i]
            cls = ids.index(tokenizer.cls_token_id)
            secuencias = entradas.sequence_ids(i)
            respuesta = ejemplos["answers"][originales[i]]

            #CLS representa que esta ventana no contiene una respuesta.
            inicio_token = cls
            final_token = cls

            if respuesta["text"]:
                inicio_char = respuesta["answer_start"][0]
                final_char = inicio_char + len(respuesta["text"][0])

                contexto = [
                    j for j, secuencia in enumerate(secuencias)
                    if secuencia == 1
                ]
                primero = contexto[0]
                ultimo = contexto[-1]

                #Solo etiquetamos la respuesta si cabe completa.
                if (
                    mapa[primero][0] <= inicio_char
                    and mapa[ultimo][1] >= final_char
                ):
                    inicio_token = primero
                    while (
                        inicio_token <= ultimo
                        and mapa[inicio_token][0] <= inicio_char
                    ):
                        inicio_token += 1
                    inicio_token -= 1

                    final_token = ultimo
                    while (
                        final_token >= primero
                        and mapa[final_token][1] >= final_char
                    ):
                        final_token -= 1
                    final_token += 1

            inicios.append(inicio_token)
            finales.append(final_token)

        entradas["start_positions"] = inicios
        entradas["end_positions"] = finales
        return entradas

    entrenamiento = muestra.map(
        preparar,
        batched=True,
        remove_columns=muestra.column_names,
    )

    argumentos = TrainingArguments(
        output_dir=str(salida),
        max_steps=20,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=5,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=argumentos,
        train_dataset=entrenamiento,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.save_model(str(salida / "modelo"))
    trainer.save_state()

    with open(salida / "historial.json", "w", encoding="utf-8") as archivo:
        json.dump(trainer.state.log_history, archivo, indent=2)

    memoria = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\nMáxima memoria GPU asignada por PyTorch: {memoria:.2f} GiB")
    print("Prueba terminada. Modelo guardado en outputs/prueba/modelo")


if __name__ == "__main__":
    main()