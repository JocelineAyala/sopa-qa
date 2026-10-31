"""Diagnostica V2 en las 1000 preguntas de seguimiento, sin entrenar."""

# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from qa_v3 import infer_rows, measure


def main():
    root = ROOT
    logs = root / 'logs/roberta_squad2_v3'
    config = json.loads((logs / 'umbral.json').read_text(encoding='utf-8'))
    if config.get('source') != 'parent_v2':
        raise RuntimeError('El modelo seleccionado ya no es V2. Revisar antes de analizar.')
    path = root / config['model_path']
    threshold = float(config['null_threshold'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True)
    model.to(device)
    model.eval()
    validation = load_from_disk(str(root / 'data/squad_v2'))['validation'].shuffle(seed=42)
    examples = list(validation.select(range(1000, 2000)))
    print('Analizando V2: 1000 preguntas de seguimiento. Sin actualizar pesos ni umbral.')
    rows = infer_rows(model, tokenizer, examples)
    metrics = measure(rows, threshold)
    groups = defaultdict(list)
    counts = Counter()
    for example, row in zip(examples, rows):
        if example['id'] != row['id']:
            raise RuntimeError('Las preguntas y las predicciones no coinciden.')
        abstain = row['difference'] > threshold
        suffix = 'null' if abstain else 'span'
        if row['em_' + suffix] == 1:
            counts['aciertos'] += 1
            continue
        if row['has_answer'] and abstain:
            kind = 'rechazo_incorrecto'
        elif not row['has_answer']:
            kind = 'respuesta_en_pregunta_sin_respuesta'
        elif row['f1_span'] > 0:
            kind = 'extraccion_con_solapamiento'
        else:
            kind = 'extraccion_sin_solapamiento'
        counts[kind] += 1
        groups[kind].append(dict(
            id=row['id'], tipo=kind, pregunta=example['question'],
            contexto=example['context'], respuestas_aceptadas=row['references'],
            prediccion='' if abstain else row['span'],
            mejor_fragmento=row['span'], f1_fragmento=round(100 * row['f1_span'], 2),
            diferencia=row['difference'], umbral=threshold,
            margen_no_respuesta=row['difference'] - threshold,
        ))
    #Muestra reproducible distribuida entre tipos; los conteos usan los 1000 casos.
    rng = random.Random(42)
    for group in groups.values():
        rng.shuffle(group)
    sample = []
    positions = Counter()
    while len(sample) < 30:
        added = False
        for kind in sorted(groups):
            index = positions[kind]
            if index < len(groups[kind]) and len(sample) < 30:
                sample.append(groups[kind][index])
                positions[kind] += 1
                added = True
        if not added:
            break
    report = dict(modelo=str(path), particion='validation.shuffle(seed=42)[1000:2000]',
                  umbral=threshold, metricas=metrics, conteos=dict(counts),
                  muestra=sample,
                  nota='Solapamiento de palabras no demuestra un error de limites. Revisar el contexto. No se modificaron pesos ni umbral.')
    destination = logs / 'diagnostico_v2'
    destination.mkdir(exist_ok=True)
    (destination / 'errores_v2.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['DIAGNOSTICO V2: 1000 PREGUNTAS DE SEGUIMIENTO',
             json.dumps(metrics, ensure_ascii=False, indent=2),
             '\nCONTEOS EN LAS 1000 PREGUNTAS',
             json.dumps(dict(counts), ensure_ascii=False, indent=2),
             '\nMUESTRA DISTRIBUIDA ENTRE TIPOS; NO ESTIMA SU FRECUENCIA.',
             'El solapamiento no demuestra un error de limites; requiere revision.']
    for i, item in enumerate(sample, 1):
        lines.extend([f'\nCASO {i}: {item["tipo"]}',
                      json.dumps(item, ensure_ascii=False, indent=2)])
    (destination / 'errores_v2.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\nCONTEOS:', json.dumps(dict(counts), ensure_ascii=False))
    print('EM global:', round(metrics['overall']['EM'], 2))
    print('F1 global:', round(metrics['overall']['F1'], 2))
    print('Reporte para compartir:', destination / 'errores_v2.txt')


if __name__ == '__main__':
    main()
