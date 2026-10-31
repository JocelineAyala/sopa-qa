"""Compara posteriormente el modelo seleccionado o el candidato fusionado."""

# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from versiones.v4_lora.qa_v4 import infer_rows, measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidato', action='store_true')
    options = parser.parse_args()
    root = ROOT
    logs = root / 'logs/roberta_squad2_v4_lora'
    filename = 'umbral_candidato.json' if options.candidato else 'umbral.json'
    configuration = json.loads((logs / filename).read_text(encoding='utf-8'))
    path = root / configuration['model_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True)
    model.to(device)
    model.eval()
    data = load_from_disk(str(root / 'data/squad_v2'))['validation'].shuffle(seed=42)
    examples = list(data.select(range(2000, len(data))))
    print('Modelo:', configuration['source'])
    print('Preguntas:', len(examples))
    print('No se ajusta el umbral con estas preguntas.')
    rows = infer_rows(model, tokenizer, examples)
    result = measure(rows, configuration['null_threshold'])
    for name in ('overall', 'answerable', 'unanswerable'):
        group = result[name]
        print(f'{name}: n={group["count"]} | EM={group["EM"]:.2f} | F1={group["F1"]:.2f}')
    kind = 'candidato' if options.candidato else 'seleccionado'
    destination = logs / f'evaluacion_9873_{kind}.json'
    destination.write_text(json.dumps(dict(configuration=configuration, metrics=result),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('Resultados:', destination)
    print('Estas preguntas ya participaron en comparaciones anteriores; no son un test independiente.')


if __name__ == '__main__':
    main()
