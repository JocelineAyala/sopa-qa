"""Elige un umbral por EM en calibracion y lo compara en seguimiento."""

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


def choose_em_threshold(rows, baseline):
    """Barrido exacto de decisiones: difference > threshold significa vacio."""
    if not rows or not math.isfinite(baseline):
        raise ValueError('Se necesitan predicciones y un umbral finito.')
    if any(not math.isfinite(r['difference']) for r in rows):
        raise ValueError('Puntuaciones no finitas.')
    ordered = sorted(rows, key=lambda r: r['difference'])
    em = sum(r['em_null'] for r in rows)
    f1 = sum(r['f1_null'] for r in rows)
    first = math.nextafter(ordered[0]['difference'], -math.inf)
    #En empates de EM usamos F1; luego preferimos el umbral actual.
    def quality(e, f, t):
        return (round(e, 10), round(f, 10), -abs(t - baseline))
    best_threshold = baseline
    old_em = sum(r['em_null' if r['difference'] > baseline else 'em_span'] for r in rows)
    old_f1 = sum(r['f1_null' if r['difference'] > baseline else 'f1_span'] for r in rows)
    best = quality(old_em, old_f1, baseline)
    if quality(em, f1, first) > best:
        best, best_threshold = quality(em, f1, first), first
    i = 0
    while i < len(ordered):
        threshold = ordered[i]['difference']
        while i < len(ordered) and ordered[i]['difference'] == threshold:
            row = ordered[i]
            em += row['em_span'] - row['em_null']
            f1 += row['f1_span'] - row['f1_null']
            i += 1
        candidate = quality(em, f1, threshold)
        if candidate > best:
            best, best_threshold = candidate, threshold
    return best_threshold


def main():
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from qa_v3 import infer_rows, measure

    root = ROOT
    logs = root / 'logs/roberta_squad2_v3'
    configuration = json.loads((logs / 'umbral.json').read_text(encoding='utf-8'))
    if configuration.get('source') != 'parent_v2':
        raise RuntimeError('El modelo seleccionado ya no es V2. Revisar la configuracion.')
    baseline = float(configuration['null_threshold'])
    path = root / configuration['model_path']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Modelo:', path)
    print('Dispositivo:', device)
    print('No se entrenan pesos ni se modifica umbral.json.')
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True)
    model.to(device)
    model.eval()
    data = load_from_disk(str(root / 'data/squad_v2'))['validation'].shuffle(seed=42)
    examples = list(data.select(range(2000)))
    rows = infer_rows(model, tokenizer, examples)
    if [e['id'] for e in examples] != [r['id'] for r in rows]:
        raise RuntimeError('Las predicciones no coinciden con las preguntas.')
    calibration, tracking = rows[:1000], rows[1000:]
    #El selector recibe exclusivamente calibracion, nunca seguimiento.
    candidate = choose_em_threshold(calibration, baseline)
    results = {}
    for name, partition in [('calibracion', calibration), ('seguimiento', tracking)]:
        results[name] = dict(actual=measure(partition, baseline),
                             candidato=measure(partition, candidate))
    report = dict(modelo=str(path), umbral_actual=baseline, umbral_candidato=candidate,
                  objetivo='EM global; desempate F1; luego cercania al umbral actual',
                  particiones=dict(calibracion=[0, 1000], seguimiento=[1000, 2000], seed=42),
                  resultados=results,
                  nota='Particiones de desarrollo reutilizadas. No es un test independiente. No se cambia la API.')
    destination = logs / 'comparacion_umbrales_v2'
    destination.mkdir(exist_ok=True)
    (destination / 'comparacion.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    candidate_config = dict(configuration)
    candidate_config.update(null_threshold=candidate, calibration_objective='overall_EM')
    (destination / 'umbral_candidato_em.json').write_text(
        json.dumps(candidate_config, ensure_ascii=False, indent=2), encoding='utf-8')
    (destination / 'predicciones_desarrollo.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['COMPARACION DE UMBRALES V2', f'Umbral actual: {baseline:.8f}',
             f'Umbral candidato: {candidate:.8f}',
             'Candidato elegido SOLO con calibracion; no se cambio la configuracion actual.']
    for name, values in results.items():
        lines.append('\n' + name.upper())
        for version, metrics in values.items():
            lines.append(f'{version}: EM global {metrics["overall"]["EM"]:.2f} | '
                         f'F1 global {metrics["overall"]["F1"]:.2f} | '
                         f'EM con respuesta {metrics["answerable"]["EM"]:.2f} | '
                         f'EM sin respuesta {metrics["unanswerable"]["EM"]:.2f}')
    lines.append('\nLas particiones de desarrollo ya se han reutilizado; una mejora aqui no garantiza mejora externa.')
    text = '\n'.join(lines)
    (destination / 'comparacion.txt').write_text(text, encoding='utf-8')
    print('\n' + text)
    print('\nReporte:', destination / 'comparacion.txt')


if __name__ == '__main__':
    main()
