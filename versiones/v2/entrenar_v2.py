
# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

from transformers import EarlyStoppingCallback
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from transformers.trainer_utils import get_last_checkpoint


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    set_seed(42)

    salida = ROOT / "outputs/roberta_squad2_v2"
    registros = ROOT / "logs/roberta_squad2_v2"
    salida.mkdir(parents=True, exist_ok=True)
    registros.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))

    #Cargamos el modelo base y el tokenizador descargados.
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "models/roberta-base"),
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        str(ROOT / "models/roberta-base")
    )

    #Usamos todos los ejemplos de entrenamiento de SQuAD 2.0.
    datos = load_from_disk(str(ROOT / "data/squad_v2"))
    muestra = datos["train"].shuffle(seed=42)

    def preparar(ejemplos):
        #Dividimos los contextos largos en ventanas solapadas.
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

            #CLS indica que esta ventana no contiene una respuesta.
            inicio_token = cls
            final_token = cls

            contexto = [
                j for j, secuencia in enumerate(secuencias)
                if secuencia == 1
            ]

            if respuesta["text"] and contexto:
                inicio_char = respuesta["answer_start"][0]
                final_char = inicio_char + len(respuesta["text"][0])

                primero = contexto[0]
                ultimo = contexto[-1]

                #La respuesta debe estar completamente dentro de la ventana.
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

    print("Preparando datos de entrenamiento...")

    entrenamiento = muestra.map(
        preparar,
        batched=True,
        remove_columns=muestra.column_names,
    )

    #Usamos una muestra fija de validación para monitorear la pérdida.
    validacion_original = (
        datos["validation"]
        .shuffle(seed=42)
        .select(range(2000))
    )

    print("Preparando datos de validación...")

    validacion = validacion_original.map(
        preparar,
        batched=True,
        remove_columns=validacion_original.column_names,
    )

    print("Ejemplos originales de entrenamiento:", len(muestra))
    print("Ventanas de entrenamiento:", len(entrenamiento))
    print("Ejemplos originales de validación:", len(validacion_original))
    print("Ventanas de validación:", len(validacion))

    argumentos = TrainingArguments(
        output_dir=str(salida),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_steps=824,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
    )

    #Guardamos la configuración antes de empezar.
    configuracion = {
        "base_checkpoint": "FacebookAI/roberta-base",
        "training_examples": len(muestra),
        "training_windows": len(entrenamiento),
        "monitoring_validation_examples": len(validacion_original),
        "validation_windows": len(validacion),
        "max_seq_length": 384,
        "doc_stride": 128,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "training_arguments": argumentos.to_dict(),
    }

    with open(
        registros / "configuracion.json",
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(configuracion, archivo, indent=2, default=str)

    trainer = Trainer(
        model=model,
        args=argumentos,
        train_dataset=entrenamiento,
        eval_dataset=validacion,
        data_collator=default_data_collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=6
        )],
    )

    #Retomamos el último checkpoint completo si existe.
    checkpoint = get_last_checkpoint(str(salida))

    if checkpoint:
        print("Reanudando desde:", checkpoint)
    else:
        print("Iniciando desde el modelo base descargado.")

    torch.cuda.reset_peak_memory_stats()

    try:
        resultado = trainer.train(
            resume_from_checkpoint=checkpoint
        )

        #Trainer recupera el checkpoint con menor pérdida de validación.
        trainer.save_model(str(salida / "modelo"))
        trainer.save_state()
        trainer.save_metrics("train", resultado.metrics)

        resumen = {
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_loss": trainer.state.best_metric,
            "global_step": trainer.state.global_step,
            "training_metrics": resultado.metrics,
        }

        with open(
            registros / "resumen.json",
            "w",
            encoding="utf-8",
        ) as archivo:
            json.dump(resumen, archivo, indent=2, default=str)

        print("\nEntrenamiento terminado.")
        print("Modelo guardado en:", salida / "modelo")
        print(
            "Checkpoint con menor pérdida:",
            trainer.state.best_model_checkpoint,
        )

    finally:
        #Guardamos los registros también ante una interrupción normal.
        historial = trainer.state.log_history

        with open(
            registros / "historial.json",
            "w",
            encoding="utf-8",
        ) as archivo:
            json.dump(historial, archivo, indent=2, default=str)

        entrenamiento_logs = [
            x for x in historial if "loss" in x
        ]

        validacion_logs = [
            x for x in historial if "eval_loss" in x
        ]

        if entrenamiento_logs or validacion_logs:
            plt.figure(figsize=(9, 5))

            if entrenamiento_logs:
                plt.plot(
                    [x["step"] for x in entrenamiento_logs],
                    [float(x["loss"]) for x in entrenamiento_logs],
                    label="Training loss",
                )

            if validacion_logs:
                plt.plot(
                    [x["step"] for x in validacion_logs],
                    [float(x["eval_loss"]) for x in validacion_logs],
                    label="Validation loss",
                )

            plt.xlabel("Optimizer steps")
            plt.ylabel("Loss")
            plt.title("RoBERTa-base fine-tuning on SQuAD 2.0")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(registros / "loss.png", dpi=160)
            plt.close()

        memoria = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

        print(
            f"\nMáxima memoria GPU asignada por PyTorch: "
            f"{memoria:.2f} GiB"
        )
        print("Registros guardados en:", registros)


if __name__ == "__main__":
    main()