"""Prueba corta con errores de TRAIN. Conserva V2 y no modifica la API.
Ejecutar desde la raíz: .\\.venv\\Scripts\\python.exe versiones/v5/entrenar_v5_rapido.py
Requiere qa_v3.py, que ya se utilizó en V3. --evaluar evalúa la selección final.
"""

# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from qa_v3 import (tokenize_examples, decode, select_threshold, measure,
                   infer_rows, should_promote)

PARENT = ROOT / 'outputs/roberta_squad2_v2/modelo'
PARENT_CONFIG = ROOT / 'logs/roberta_squad2_v2/umbral.json'
OUTPUT = ROOT / 'outputs/roberta_squad2_v5_rapido'
LOGS = ROOT / 'logs/roberta_squad2_v5_rapido'


def save(name, value):
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def make_training_mix(pool, rows, threshold, seed=52):
    """Balancea los tipos de errores sin tomar ningún ejemplo de validación."""
    if len(pool) != len(rows):
        raise ValueError('No coincide la muestra con las predicciones.')
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, (example, row) in enumerate(zip(pool, rows)):
        if example['id'] != row['id']:
            raise ValueError('Orden de preguntas incorrecto.')
        abstain = row['difference'] > threshold
        em = row['em_null' if abstain else 'em_span']
        if em == 1:
            continue
        kind = ('rechazo_incorrecto' if row['has_answer'] and abstain else
                'respuesta_injustificada' if not row['has_answer'] else
                'extraccion')
        groups[kind].append(i)
    #No ordenamos por confianza: podría concentrar errores de anotación extremos.
    for values in groups.values():
        rng.shuffle(values)
    hard = []
    for kind in sorted(groups):
        hard.extend(groups[kind][:200])
    #2000 ejemplos aleatorios mantienen preguntas normales con y sin respuesta.
    normal = rng.sample(range(len(pool)), min(2000, len(pool)))
    indices = normal + hard + hard
    rng.shuffle(indices)
    return [pool[i] for i in indices], dict(
        errors_by_type={k: len(v) for k, v in groups.items()},
        hard_unique_count=len(hard), hard_ids=[pool[i]['id'] for i in hard],
        normal_ids=[pool[i]['id'] for i in normal],
        ordered_training_ids=[pool[i]['id'] for i in indices],
        total_examples_with_repetition=len(indices), seed=seed)


def configuration(path, threshold, source):
    return dict(model_path=str(path.relative_to(ROOT)), null_threshold=float(threshold),
                null_rule='return null if difference > null_threshold',
                difference='minimum CLS score minus best span score',
                max_seq_length=384, doc_stride=128, max_answer_length=30,
                top_k=20, source=source, threshold_calibration_count=1000,
                monitoring_count=1000)


def quality_metrics(prediction, examples, metadata, baseline):
    starts, ends = prediction.predictions[:2]
    rows = decode(examples, metadata, starts, ends)
    threshold = select_threshold(rows[:1000])
    quality = measure(rows[1000:], threshold)
    eligible = should_promote(quality, baseline)
    a, n = quality['answerable']['EM'], quality['unanswerable']['EM']
    print(f'\nQA: con respuesta {a:.2f}% | sin respuesta {n:.2f}% | '
          f'armonico {quality["harmonic_EM"]:.2f}% | cumple mejora: {eligible}')
    #Un checkpoint que empeora alguna clase nunca desplaza a uno elegible.
    return dict(selection_score=quality['harmonic_EM'] if eligible else -1.0,
                harmonic_em=quality['harmonic_EM'], answerable_em=a,
                unanswerable_em=n, null_threshold=threshold,
                qa_em=quality['overall']['EM'], qa_f1=quality['overall']['F1'])


def plot(history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    train = [r for r in history if 'loss' in r]
    valid = [r for r in history if 'eval_loss' in r]
    if train or valid:
        plt.figure(figsize=(8, 4))
        for records, key, label in [(train, 'loss', 'Train'), (valid, 'eval_loss', 'Validation')]:
            if records:
                plt.plot([r['step'] for r in records], [float(r[key]) for r in records], label=label)
        plt.xlabel('Optimizer steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(LOGS / 'loss.png', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluar', action='store_true', help='Evalúa la selección en 9873 preguntas; no entrena.')
    options = parser.parse_args()
    import torch
    from datasets import load_from_disk
    from transformers import (AutoTokenizer, AutoModelForQuestionAnswering,
        Trainer, TrainingArguments, EarlyStoppingCallback, default_data_collator, set_seed)
    from transformers.trainer_utils import get_last_checkpoint

    if not options.evaluar and not torch.cuda.is_available():
        raise RuntimeError('CUDA no está disponible; no inicies este entrenamiento en CPU.')
    set_seed(52)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Dispositivo:', device)
    data = load_from_disk(str(ROOT / 'data/squad_v2'))
    validation = data['validation'].shuffle(seed=42)

    if options.evaluar:
        config = json.loads((LOGS / 'umbral.json').read_text(encoding='utf-8'))
        path = ROOT / config['model_path']
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        model = AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True).to(device)
        rows = infer_rows(model, tokenizer, list(validation.select(range(2000, len(validation)))))
        metrics = measure(rows, config['null_threshold'])
        save('metricas_evaluacion.json', dict(configuration=config, metrics=metrics,
             note='Evaluación ya utilizada en comparaciones anteriores; no es test independiente.'))
        save('predicciones_evaluacion.json', rows)
        print(json.dumps(metrics, indent=2))
        return

    if (LOGS / 'resumen.json').exists():
        raise RuntimeError('Esta prueba ya terminó. Conserva sus resultados; no la repitas. Usa --evaluar.')
    if not PARENT.is_dir() or not PARENT_CONFIG.is_file():
        raise FileNotFoundError('Falta V2 exportada o su umbral.json.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    checkpoint = get_last_checkpoint(str(OUTPUT))
    parent_config = json.loads(PARENT_CONFIG.read_text(encoding='utf-8'))
    parent_threshold = float(parent_config['null_threshold'])
    tokenizer = AutoTokenizer.from_pretrained(PARENT, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(PARENT, local_files_only=True).to(device)
    development = list(validation.select(range(2000)))
    pool = list(data['train'].shuffle(seed=52).select(range(min(3000, len(data['train'])))))
    if {e['id'] for e in pool} & set(validation['id']):
        raise RuntimeError('Se detectaron identificadores compartidos entre train y validation.')
    print('Buscando errores de V2 en 3000 preguntas de TRAIN...')
    cache = LOGS / 'mineria_train.json'
    if cache.exists():
        cached = json.loads(cache.read_text(encoding='utf-8'))
        if (cached['pool_ids'] != [e['id'] for e in pool]
                or cached['threshold'] != parent_threshold):
            raise RuntimeError('La caché no coincide con esta selección o umbral.')
        rows = cached['rows']
    else:
        rows = infer_rows(model, tokenizer, pool)
        save('mineria_train.json', dict(pool_ids=[e['id'] for e in pool],
             threshold=parent_threshold, rows=rows))
    mixed, mining = make_training_mix(pool, rows, parent_threshold)
    save('muestra_entrenamiento.json', mining)
    print('Errores encontrados:', mining['errors_by_type'])
    print('Preguntas de entrenamiento contando repeticiones:', len(mixed))
    print('Preparando ventanas...')
    train_features, _ = tokenize_examples(mixed, tokenizer)
    eval_features, metadata = tokenize_examples(development, tokenizer)
    print('Midiendo V2 como referencia...')
    parent_rows = infer_rows(model, tokenizer, development)
    baseline = measure(parent_rows[1000:], parent_threshold)
    save('baseline_v2.json', dict(threshold=parent_threshold, metrics=baseline))
    save('particiones.json', dict(calibration_ids=[e['id'] for e in development[:1000]],
         monitoring_ids=[e['id'] for e in development[1000:]],
         note='Desarrollo reutilizado; ningún ejemplo de validation se usa para actualizar pesos.'))
    arguments = TrainingArguments(
        output_dir=str(OUTPUT), max_steps=600,
        per_device_train_batch_size=2, per_device_eval_batch_size=2,
        gradient_accumulation_steps=4, learning_rate=2e-6,
        warmup_steps=30, weight_decay=0.01, lr_scheduler_type='linear',
        fp16=True, gradient_checkpointing=True, optim='adamw_torch',
        logging_steps=50, eval_strategy='steps', eval_steps=200,
        save_strategy='steps', save_steps=200, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='selection_score',
        greater_is_better=True, prediction_loss_only=False,
        eval_accumulation_steps=32, label_names=['start_positions', 'end_positions'],
        dataloader_num_workers=0, report_to='none', seed=52)
    save('configuracion.json', dict(base_checkpoint='FacebookAI/roberta-base',
         continued_from=str(PARENT), mining_pool_count=len(pool),
         training_windows=len(train_features), training_arguments=arguments.to_dict(),
         torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(0)))
    trainer = Trainer(model=model, args=arguments, train_dataset=train_features,
        eval_dataset=eval_features, data_collator=default_data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda p: quality_metrics(p, development, metadata, baseline),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    print('Iniciando: máximo 600 pasos; V2 y la API permanecen sin cambios.')
    torch.cuda.reset_peak_memory_stats()
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()
        candidate_path = OUTPUT / 'candidato'
        trainer.save_model(str(candidate_path))
        tokenizer.save_pretrained(candidate_path)
        #Reevaluamos en precisión completa antes de seleccionar.
        candidate_rows = infer_rows(trainer.model, tokenizer, development)
        threshold = select_threshold(candidate_rows[:1000])
        candidate = measure(candidate_rows[1000:], threshold)
        save('predicciones_desarrollo_candidato.json', candidate_rows)
        save('umbral_candidato.json', configuration(candidate_path, threshold, 'v5_candidate'))
        improved = should_promote(candidate, baseline)
        chosen_path = candidate_path if improved else PARENT
        chosen_threshold = threshold if improved else parent_threshold
        selected = 'v5_candidate' if improved else 'parent_v2'
        save('umbral.json', configuration(chosen_path, chosen_threshold, selected))
        save('resumen.json', dict(selected=selected, baseline_v2=baseline,
             candidate=candidate, selected_threshold=chosen_threshold,
             best_candidate_checkpoint=trainer.state.best_model_checkpoint,
             global_step=trainer.state.global_step, training_metrics=result.metrics,
             note='Solo se selecciona con desarrollo reutilizado; API sin modificar.'))
        print('\nSELECCION:', selected)
        print('V2:', json.dumps(baseline, ensure_ascii=False))
        print('Candidato:', json.dumps(candidate, ensure_ascii=False))
        print('Resumen:', LOGS / 'resumen.json')
        print('No cambies la API antes de revisar estos resultados.')
    finally:
        save('historial.json', trainer.state.log_history)
        plot(trainer.state.log_history)
        print(f'Máxima memoria GPU asignada: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB')


if __name__ == '__main__':
    main()
