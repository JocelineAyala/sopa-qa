"""Continúa nuestra V2 y selecciona checkpoints con métricas QA por clase."""

# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import json
import math
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering, Trainer,
    TrainingArguments, EarlyStoppingCallback, TrainerCallback,
    default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from qa_v3 import (tokenize_examples, decode, select_threshold, measure, infer_rows, should_promote)

PARENT = ROOT / "outputs/roberta_squad2_v2/modelo"
PARENT_CONFIG = ROOT / "logs/roberta_squad2_v2/umbral.json"
OUTPUT = ROOT / "outputs/roberta_squad2_v3"
LOGS = ROOT / "logs/roberta_squad2_v3"


def save(name, data):
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class QAMetrics:
    def __init__(self, examples, metadata):
        self.examples, self.metadata = examples, metadata

    def __call__(self, prediction):
        starts, ends = prediction.predictions[:2]
        rows = decode(self.examples, self.metadata, starts, ends)
        threshold = select_threshold(rows[:1000])
        quality = measure(rows[1000:], threshold)
        a, n = quality["answerable"], quality["unanswerable"]
        print(f"\nQA seguimiento | EM con respuesta: {a['EM']:.2f} | "
              f"EM sin respuesta: {n['EM']:.2f} | "
              f"Media armonica EM: {quality['harmonic_EM']:.2f} | "
              f"Umbral: {threshold:.4f}")
        return dict(harmonic_em=quality["harmonic_EM"],
                    answerable_em=a["EM"], unanswerable_em=n["EM"],
                    qa_em=quality["overall"]["EM"], qa_f1=quality["overall"]["F1"],
                    null_threshold=threshold)


class RecordQuality(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        record = dict(step=state.global_step, epoch=state.epoch, metrics=metrics)
        with (LOGS / "calidad_por_checkpoint.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def export_config(threshold, model_path, source, selected_on):
    return dict(model_path=str(model_path.relative_to(ROOT)), null_threshold=float(threshold),
                null_rule="return null if difference > null_threshold",
                difference="minimum CLS score minus best span score",
                max_seq_length=384, doc_stride=128, max_answer_length=30, top_k=20,
                selection_metric=selected_on, source=source,
                threshold_calibration_count=1000, monitoring_count=1000)


def plot_history(history):
    train = [r for r in history if "loss" in r]
    valid = [r for r in history if "eval_loss" in r]
    if train or valid:
        plt.figure(figsize=(9, 5))
        for rows, key, label in [(train, "loss", "Training loss"), (valid, "eval_loss", "Validation loss")]:
            if rows:
                plt.plot([r["step"] for r in rows], [float(r[key]) for r in rows], label=label)
        plt.xlabel("Optimizer steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(LOGS / "loss.png", dpi=160)
        plt.close()
    quality = [r for r in history if "eval_harmonic_em" in r]
    if quality:
        plt.figure(figsize=(9, 5))
        for key, label in [("eval_answerable_em", "Answerable EM"),
                           ("eval_unanswerable_em", "Unanswerable EM"),
                           ("eval_harmonic_em", "Harmonic mean EM")]:
            plt.plot([r["step"] for r in quality], [float(r[key]) for r in quality], label=label)
        plt.xlabel("Optimizer steps")
        plt.ylabel("Score (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        plt.savefig(LOGS / "calidad.png", dpi=160)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comprobar", action="store_true", help="Comprueba carga y evaluación sin entrenar.")
    options = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")
    if not PARENT.is_dir() or not PARENT_CONFIG.is_file():
        raise FileNotFoundError("Se necesita la V2 exportada y logs/roberta_squad2_v2/umbral.json.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    checkpoint = get_last_checkpoint(str(OUTPUT))
    if not checkpoint and (OUTPUT / "candidato").exists():
        raise RuntimeError("Esta V3 ya tiene un candidato exportado. Conservá sus resultados antes de repetirla.")
    set_seed(42)
    print("GPU:", torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(PARENT, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(PARENT, local_files_only=True)
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

    print("Preparando ventanas de entrenamiento...")
    entrenamiento = muestra.map(preparar, batched=True, remove_columns=muestra.column_names)
    #Solo usamos los 2000 ejemplos ya utilizados para desarrollo.
    development = list(datos["validation"].shuffle(seed=42).select(range(2000)))
    calibration_ids = {e["id"] for e in development[:1000]}
    monitoring_ids = {e["id"] for e in development[1000:]}
    assert calibration_ids.isdisjoint(monitoring_ids)
    save("particiones.json", dict(calibration_ids=sorted(calibration_ids), monitoring_ids=sorted(monitoring_ids)))
    print("Preparando 1000 preguntas de calibración y 1000 de seguimiento...")
    validation, metadata = tokenize_examples(development, tokenizer)
    steps = math.ceil(len(entrenamiento) / 16)
    arguments = TrainingArguments(
        output_dir=str(OUTPUT), num_train_epochs=1,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=3e-6,
        warmup_steps=max(1, round(0.05 * steps)), weight_decay=0.01,
        lr_scheduler_type="linear", fp16=True, gradient_checkpointing=True,
        optim="adamw_torch", logging_steps=100,
        eval_strategy="steps", eval_steps=500,
        prediction_loss_only=False, eval_accumulation_steps=32,
        save_strategy="steps", save_steps=500, save_total_limit=3,
        load_best_model_at_end=True, metric_for_best_model="harmonic_em",
        greater_is_better=True, label_names=["start_positions", "end_positions"],
        dataloader_num_workers=0, report_to="none", seed=42,
    )
    save("configuracion.json", dict(base_checkpoint="FacebookAI/roberta-base",
         continued_from=str(PARENT), training_examples=len(muestra),
         training_windows=len(entrenamiento), max_seq_length=384, doc_stride=128,
         calibration_count=1000, monitoring_count=1000,
         gpu=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
         training_arguments=arguments.to_dict()))
    trainer = Trainer(model=model, args=arguments,
                      train_dataset=entrenamiento, eval_dataset=validation,
                      data_collator=default_data_collator, processing_class=tokenizer,
                      compute_metrics=QAMetrics(development, metadata),
                      callbacks=[RecordQuality(), EarlyStoppingCallback(early_stopping_patience=4)])
    #Medimos la V2 con su umbral vigente en las mismas preguntas de seguimiento.
    parent_config = json.loads(PARENT_CONFIG.read_text(encoding="utf-8"))
    print("Midiendo la V2 antes de continuar el entrenamiento...")
    parent_rows = infer_rows(trainer.model, tokenizer, development)
    baseline = measure(parent_rows[1000:], parent_config["null_threshold"])
    save("baseline_v2.json", dict(threshold=parent_config["null_threshold"], metrics=baseline))
    print(f"V2 seguimiento: EM con respuesta {baseline['answerable']['EM']:.2f}; "
          f"EM sin respuesta {baseline['unanswerable']['EM']:.2f}")
    print("Comprobando la integración de métricas con Trainer...")
    preflight = trainer.evaluate()
    required = {"eval_harmonic_em", "eval_answerable_em", "eval_unanswerable_em", "eval_null_threshold"}
    if not required.issubset(preflight):
        raise RuntimeError("Trainer no devolvió las métricas QA necesarias.")
    if not all(math.isfinite(float(preflight[key])) for key in required):
        raise RuntimeError("Se encontraron métricas no finitas.")
    if options.comprobar:
        print("COMPROBACION CORRECTA. No se actualizó ningún peso.")
        return
    print("Reanudando desde: " + checkpoint if checkpoint else "Continuando desde nuestros pesos V2.")
    torch.cuda.reset_peak_memory_stats()
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        #El candidato es el mejor checkpoint por calidad QA, no por pérdida.
        trainer.save_model(str(OUTPUT / "candidato"))
        tokenizer.save_pretrained(OUTPUT / "candidato")
        print("Evaluando el candidato en precisión completa...")
        candidate_rows = infer_rows(trainer.model, tokenizer, development)
        threshold = select_threshold(candidate_rows[:1000])
        candidate = measure(candidate_rows[1000:], threshold)
        save("predicciones_desarrollo_candidato.json", candidate_rows)
        save("umbral_candidato.json", export_config(threshold, OUTPUT / "candidato", "v3_candidate", "Calibration harmonic mean of class EM"))
        #Solo promovemos si mejora el promedio armónico y no empeora ninguna clase.
        improved = should_promote(candidate, baseline)
        if improved:
            trainer.save_model(str(OUTPUT / "modelo"))
            chosen_threshold, selected = threshold, "v3_candidate"
            chosen_metrics = candidate
        else:
            #La copia se guarda en V3; los archivos originales de V2 no se modifican.
            parent_model = AutoModelForQuestionAnswering.from_pretrained(PARENT, local_files_only=True)
            parent_model.save_pretrained(OUTPUT / "modelo")
            del parent_model
            chosen_threshold, selected = parent_config["null_threshold"], "parent_v2"
            chosen_metrics = baseline
        tokenizer.save_pretrained(OUTPUT / "modelo")
        save("umbral.json", export_config(chosen_threshold, OUTPUT / "modelo", selected, "Monitoring EM in both classes; retain V2 if either class worsens"))
        save("resumen.json", dict(selected=selected,
             best_candidate_checkpoint=trainer.state.best_model_checkpoint,
             best_candidate_monitoring_harmonic_EM=trainer.state.best_metric,
             baseline_v2=baseline, candidate=candidate, selected_metrics=chosen_metrics,
             selected_threshold=chosen_threshold, global_step=trainer.state.global_step,
             training_metrics=result.metrics))
        print("\nEntrenamiento terminado.")
        print("Selección:", "V3 mejoró ambas clases o mantuvo una y mejoró la otra." if improved
              else "Conservamos V2: el candidato no mejoró sin empeorar alguna clase.")
        print("Modelo seleccionado:", OUTPUT / "modelo")
        print("Candidato para comparar:", OUTPUT / "candidato")
        print("V2:", json.dumps(baseline))
        print("V3 candidato:", json.dumps(candidate))
    finally:
        save("historial.json", trainer.state.log_history)
        plot_history(trainer.state.log_history)
        print(f"Memoria GPU asignada máxima: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
        print("Registros:", LOGS)


if __name__ == "__main__":
    main()
