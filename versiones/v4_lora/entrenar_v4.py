"""Experimento V4: LoRA sobre nuestra V2, con seleccion QA conservadora."""

# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

try:
    import torch
    import transformers
    import peft
    from datasets import load_from_disk
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoTokenizer, AutoModelForQuestionAnswering,
        Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback,
        default_data_collator, set_seed)
    from transformers.trainer_utils import get_last_checkpoint
except ImportError as exc:
    raise SystemExit('No se pudo importar una dependencia: ' + str(exc) +
        '\nInstala PEFT en el entorno del proyecto: .\\.venv\\Scripts\\python.exe -m pip install --no-deps peft')

from versiones.v4_lora.qa_v4 import (prepare_training_batch, tokenize_examples, decode,
                  select_threshold, measure, infer_rows, should_promote)

PARENT = ROOT / 'outputs/roberta_squad2_v2/modelo'
PARENT_CONFIG = ROOT / 'logs/roberta_squad2_v2/umbral.json'
OUTPUT = ROOT / 'outputs/roberta_squad2_v4_lora'
LOGS = ROOT / 'logs/roberta_squad2_v4_lora'


def save(name, value):
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      default=str), encoding='utf-8')


def config(threshold, path, source):
    return dict(model_path=str(path.relative_to(ROOT)), source=source,
        null_threshold=float(threshold),
        null_rule='return null if difference > null_threshold',
        difference='minimum CLS score minus best span score',
        max_seq_length=384, doc_stride=128, max_answer_length=30, top_k=20,
        threshold_calibration_count=1000, monitoring_count=1000)


class Metrics:
    def __init__(self, examples, metadata):
        self.examples, self.metadata = examples, metadata

    def __call__(self, prediction):
        starts, ends = prediction.predictions[:2]
        rows = decode(self.examples, self.metadata, starts, ends)
        threshold = select_threshold(rows[:1000])
        result = measure(rows[1000:], threshold)
        a, n = result['answerable']['EM'], result['unanswerable']['EM']
        print(f'\nV4 seguimiento | EM con respuesta {a:.2f} | '
              f'EM sin respuesta {n:.2f} | Media armonica {result["harmonic_EM"]:.2f}')
        return dict(harmonic_em=result['harmonic_EM'], answerable_em=a,
            unanswerable_em=n, qa_em=result['overall']['EM'],
            qa_f1=result['overall']['F1'], null_threshold=threshold)


class QualityLog(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        with (LOGS / 'calidad_por_checkpoint.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(step=state.global_step, epoch=state.epoch,
                                   metrics=metrics), default=str) + '\n')


class LoRATrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            groups = []
            for is_head, lr in [(False, 1e-5), (True, 1e-6)]:
                for decay in [True, False]:
                    params = [p for name, p in self.model.named_parameters()
                        if p.requires_grad and ('qa_outputs' in name) == is_head
                        and (p.ndim > 1) == decay]
                    if params:
                        groups.append(dict(params=params, lr=lr,
                                           weight_decay=0.01 if decay else 0.0))
            self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer


def trainable_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check_gradients(model, feature):
    device = next(model.parameters()).device
    batch = default_data_collator([feature])
    batch = {k: v.to(device) for k, v in batch.items()}
    before = trainable_digest(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise RuntimeError('La perdida de comprobacion no es finita.')
    loss.backward()
    lora_norm = head_norm = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError('Un peso congelado recibio gradientes: ' + name)
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError('Gradiente no finito: ' + name)
        norm = float(parameter.grad.float().abs().sum())
        if 'lora_' in name:
            lora_norm += norm
        elif 'qa_outputs' in name:
            head_norm += norm
    model.zero_grad(set_to_none=True)
    model.eval()
    if lora_norm <= 0 or head_norm <= 0 or trainable_digest(model) != before:
        raise RuntimeError('Fallo de gradientes LoRA/cabeza o se modificaron pesos.')
    print('Gradientes LoRA y cabeza QA correctos. Ningun peso actualizado.')


def plot_history(history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename, series, ylabel in [
        ('loss.png', [('loss', 'Entrenamiento'), ('eval_loss', 'Validacion')], 'Perdida'),
        ('calidad.png', [('eval_answerable_em', 'Con respuesta'),
                         ('eval_unanswerable_em', 'Sin respuesta'),
                         ('eval_harmonic_em', 'Media armonica')], 'EM (%)')]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, label in series:
            values = [r for r in history if key in r]
            if values:
                ax.plot([r['step'] for r in values],
                        [float(r[key]) for r in values], label=label)
        ax.set(xlabel='Pasos del optimizador', ylabel=ylabel)
        if ax.lines:
            ax.legend()
        fig.tight_layout()
        fig.savefig(LOGS / filename, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comprobar', action='store_true')
    options = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA no disponible. No reinstales PyTorch: revisa tu entorno CUDA.')
    if torch.cuda.device_count() < 1:
        raise RuntimeError('No se encontro una GPU.')
    if not PARENT.is_dir() or not PARENT_CONFIG.is_file():
        raise FileNotFoundError('Se necesita outputs/roberta_squad2_v2/modelo y logs/roberta_squad2_v2/umbral.json.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if not options.comprobar and (LOGS / 'resumen.json').exists():
        raise RuntimeError('V4 ya termino. Conserva sus resultados; no se repite automaticamente.')
    checkpoint = get_last_checkpoint(str(OUTPUT))
    set_seed(42)
    parent_config = json.loads(PARENT_CONFIG.read_text(encoding='utf-8'))
    baseline_threshold = float(parent_config['null_threshold'])
    tokenizer = AutoTokenizer.from_pretrained(PARENT, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(PARENT, local_files_only=True)
    if model.config.model_type != 'roberta':
        raise RuntimeError('Este experimento requiere RoBERTa.')
    #Comprobamos que insertar LoRA preserve inicialmente las salidas de V2.
    probe = tokenizer('What color is the sky?', 'The sky is blue.', return_tensors='pt')
    model.eval()
    with torch.no_grad():
        original = model(**probe)
    lora_config = LoraConfig(task_type=TaskType.QUESTION_ANS, r=8, lora_alpha=16,
        lora_dropout=0.05, target_modules=['query', 'value'],
        modules_to_save=['qa_outputs'], bias='none', init_lora_weights=True)
    model = get_peft_model(model, lora_config)
    model.eval()
    with torch.no_grad():
        adapted = model(**probe)
    if not all(torch.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in [
        (original.start_logits, adapted.start_logits), (original.end_logits, adapted.end_logits)]):
        raise RuntimeError('La insercion de LoRA altero las salidas iniciales de V2.')
    del original, adapted
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not ('lora_' in name or 'qa_outputs' in name):
            raise RuntimeError('Parametro entrenable inesperado: ' + name)
    model.print_trainable_parameters()
    versions = dict(python=platform.python_version(), torch=str(torch.__version__),
        transformers=transformers.__version__, peft=peft.__version__,
        gpu=torch.cuda.get_device_name(0))
    print('Versiones:', json.dumps(versions))
    data = load_from_disk(str(ROOT / 'data/squad_v2'))
    training_source = data['train'].shuffle(seed=42)
    print('Preparando ventanas de entrenamiento...')
    training = training_source.map(lambda batch: prepare_training_batch(batch, tokenizer),
        batched=True, remove_columns=training_source.column_names)
    development = list(data['validation'].shuffle(seed=42).select(range(2000)))
    calibration_ids = {e['id'] for e in development[:1000]}
    monitoring_ids = {e['id'] for e in development[1000:]}
    if not calibration_ids.isdisjoint(monitoring_ids):
        raise RuntimeError('Las particiones se solapan.')
    save('particiones.json', dict(calibration_ids=sorted(calibration_ids),
                                  monitoring_ids=sorted(monitoring_ids)))
    validation, metadata = tokenize_examples(development, tokenizer)
    steps = math.ceil(len(training) / 16)
    arguments = TrainingArguments(output_dir=str(OUTPUT), num_train_epochs=1,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=8, learning_rate=1e-5,
        warmup_steps=max(1, round(0.05 * steps)), weight_decay=0.01,
        lr_scheduler_type='linear', fp16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='adamw_torch', logging_steps=100,
        eval_strategy='steps', eval_steps=500, save_strategy='steps', save_steps=500,
        save_total_limit=3, load_best_model_at_end=True,
        metric_for_best_model='harmonic_em', greater_is_better=True,
        label_names=['start_positions', 'end_positions'],
        eval_accumulation_steps=32, prediction_loss_only=False,
        dataloader_num_workers=0, report_to='none', seed=42)
    save('configuracion.json', dict(versions=versions, continued_from=str(PARENT),
        base_checkpoint='FacebookAI/roberta-base', lora=lora_config.to_dict(),
        learning_rate_lora=1e-5, learning_rate_head=1e-6,
        training_examples=len(training_source), training_windows=len(training),
        training_arguments=arguments.to_dict(),
        note='Particiones de desarrollo reutilizadas; test oculto externo.'))
    trainer = LoRATrainer(model=model, args=arguments, train_dataset=training,
        eval_dataset=validation, data_collator=default_data_collator,
        processing_class=tokenizer, compute_metrics=Metrics(development, metadata),
        callbacks=[QualityLog(), EarlyStoppingCallback(early_stopping_patience=4)])
    check_gradients(trainer.model, training[0])
    #Sin entrenar, LoRA equivale a V2; la cabeza conserva los pesos originales.
    print('Midiendo V2 con su umbral actual, en precision completa...')
    parent_rows = infer_rows(trainer.model, tokenizer, development)
    baseline = measure(parent_rows[1000:], baseline_threshold)
    save('baseline_v2.json', dict(threshold=baseline_threshold, metrics=baseline))
    print('V2 seguimiento:', json.dumps(baseline))
    print('Comprobando integracion con Trainer...')
    preflight = trainer.evaluate()
    required = ['eval_harmonic_em', 'eval_answerable_em', 'eval_unanswerable_em',
                'eval_null_threshold']
    if not all(key in preflight and math.isfinite(float(preflight[key])) for key in required):
        raise RuntimeError('Fallo en las metricas QA de Trainer.')
    if options.comprobar:
        print('COMPROBACION CORRECTA. Pesos congelados, gradientes LoRA/cabeza y metricas QA verificados. No se actualizo ningun peso.')
        return
    print('Reanudando: ' + checkpoint if checkpoint else 'Entrenando LoRA sobre V2.')
    torch.cuda.reset_peak_memory_stats()
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()
        trainer.save_model(str(OUTPUT / 'adaptador_candidato'))
        tokenizer.save_pretrained(OUTPUT / 'adaptador_candidato')
        trainer.model.eval()
        trainer.model.gradient_checkpointing_disable()
        check_inputs = {k: v.to(next(trainer.model.parameters()).device) for k, v in probe.items()}
        with torch.inference_mode():
            unmerged = trainer.model(**check_inputs)
            expected = (unmerged.start_logits.float().cpu(), unmerged.end_logits.float().cpu())
        #La exportacion se fusiona para cargarla como RoBERTa sin PEFT en la API.
        merged = trainer.model.merge_and_unload(safe_merge=True)
        merged.eval()
        with torch.inference_mode():
            actual = merged(**check_inputs)
        if not all(torch.allclose(a, b.float().cpu(), atol=1e-4, rtol=1e-4)
                   for a, b in zip(expected, (actual.start_logits, actual.end_logits))):
            raise RuntimeError('La fusion LoRA no preservo las salidas. Adaptador guardado para revision.')
        merged.save_pretrained(OUTPUT / 'candidato')
        tokenizer.save_pretrained(OUTPUT / 'candidato')
        print('Evaluando candidato V4 fusionado, en precision completa...')
        candidate_rows = infer_rows(merged, tokenizer, development)
        threshold = select_threshold(candidate_rows[:1000])
        candidate = measure(candidate_rows[1000:], threshold)
        save('predicciones_desarrollo_candidato.json', candidate_rows)
        save('umbral_candidato.json', config(threshold, OUTPUT / 'candidato', 'v4_lora_candidate'))
        improved = should_promote(candidate, baseline)
        if improved:
            merged.save_pretrained(OUTPUT / 'modelo')
            chosen_threshold, selected, chosen = threshold, 'v4_lora_candidate', candidate
        else:
            original_model = AutoModelForQuestionAnswering.from_pretrained(PARENT, local_files_only=True)
            original_model.save_pretrained(OUTPUT / 'modelo')
            del original_model
            chosen_threshold, selected, chosen = baseline_threshold, 'parent_v2', baseline
        tokenizer.save_pretrained(OUTPUT / 'modelo')
        save('umbral.json', config(chosen_threshold, OUTPUT / 'modelo', selected))
        save('resumen.json', dict(selected=selected, baseline_v2=baseline,
            candidate=candidate, selected_metrics=chosen, selected_threshold=chosen_threshold,
            best_candidate_checkpoint=trainer.state.best_model_checkpoint,
            global_step=trainer.state.global_step, training_metrics=result.metrics))
        print('\nENTRENAMIENTO V4 TERMINADO.')
        print('Seleccion:', selected)
        print('V2:', json.dumps(baseline))
        print('V4 candidato:', json.dumps(candidate))
        print('Modelo seleccionado:', OUTPUT / 'modelo')
        print('No se modifico la API ni las carpetas V2/V3.')
    finally:
        save('historial.json', trainer.state.log_history)
        plot_history(trainer.state.log_history)
        print(f'Memoria GPU asignada maxima: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB')
        print('Registros:', LOGS)


if __name__ == '__main__':
    main()
