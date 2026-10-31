"""Evalúa el modelo seleccionado por V3 con un umbral ya fijado."""

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
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from qa_v3 import infer_rows, measure

LOGS = ROOT / 'logs/roberta_squad2_v3'


def main():
    configuration = json.loads((LOGS / 'umbral.json').read_text(encoding='utf-8'))
    path = ROOT / configuration['model_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True)
    model.to(device)
    model.eval()
    validation = load_from_disk(str(ROOT / 'data/squad_v2'))['validation'].shuffle(seed=42)
    #No usamos estas preguntas para elegir umbral ni checkpoint en esta corrida.
    examples = list(validation.select(range(2000, len(validation))))
    print('Modelo seleccionado:', configuration['source'])
    print('Umbral fijado:', configuration['null_threshold'])
    print('Preguntas para comparar:', len(examples))
    rows = infer_rows(model, tokenizer, examples)
    metrics = measure(rows, configuration['null_threshold'])
    for name in ('overall', 'answerable', 'unanswerable'):
        value = metrics[name]
        print(f"{name}: n={value['count']} | EM={value['EM']:.2f} | F1={value['F1']:.2f}")
    print(f"Media armonica de EM por clase: {metrics['harmonic_EM']:.2f}")
    (LOGS / 'metricas_evaluacion.json').write_text(json.dumps(
        dict(configuration=configuration, metrics=metrics), ensure_ascii=False, indent=2), encoding='utf-8')
    (LOGS / 'predicciones_evaluacion.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Resultados:', LOGS / 'metricas_evaluacion.json')


if __name__ == '__main__':
    main()
