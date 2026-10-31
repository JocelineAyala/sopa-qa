"""Recupera el adaptador V4 guardado y finaliza la evaluacion sin entrenar."""

# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT

import gc
import json
from pathlib import Path


def verify_logits(expected, actual, atol=1e-3, rtol=1e-4):
    """Compara tensores finitos y registra la diferencia; nunca omite la prueba."""
    import torch
    report = dict(atol=atol, rtol=rtol, tensors=[])
    for left, right in zip(expected, actual):
        left, right = left.detach().float().cpu(), right.detach().float().cpu()
        if left.shape != right.shape:
            report['tensors'].append(dict(passed=False, reason='shape mismatch'))
            continue
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        delta = (left - right).abs()
        report['tensors'].append(dict(passed=finite and bool(torch.allclose(
            left, right, atol=atol, rtol=rtol)), finite=finite,
            max_abs_difference=float(delta.max()), mean_abs_difference=float(delta.mean())))
    report['passed'] = len(report['tensors']) == 2 and all(
        item['passed'] for item in report['tensors'])
    return report


def finalize(root=None):
    import torch
    import peft
    import transformers
    from datasets import load_from_disk
    from peft import PeftModel
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from versiones.v4_lora.qa_v4 import infer_rows, select_threshold, measure, should_promote

    root = Path(root) if root is not None else ROOT
    parent = root / 'outputs/roberta_squad2_v2/modelo'
    parent_config_path = root / 'logs/roberta_squad2_v2/umbral.json'
    output = root / 'outputs/roberta_squad2_v4_lora'
    logs = root / 'logs/roberta_squad2_v4_lora'
    adapter = output / 'adaptador_candidato'

    def save(name, value):
        (logs / name).write_text(json.dumps(value, ensure_ascii=False,
            indent=2, default=str), encoding='utf-8')

    if (logs / 'resumen.json').is_file():
        print('V4 ya esta finalizada. Resumen:', logs / 'resumen.json')
        return
    required = [adapter / 'adapter_config.json', logs / 'baseline_v2.json', parent_config_path]
    if not all(path.is_file() for path in required) or not parent.is_dir():
        raise FileNotFoundError('Falta el adaptador guardado, la referencia V2 o su umbral. No se inicia entrenamiento.')
    if not any((adapter / name).is_file() for name in ['adapter_model.safetensors', 'adapter_model.bin']):
        raise FileNotFoundError('No se encontraron pesos del adaptador guardado.')
    baseline_record = json.loads((logs / 'baseline_v2.json').read_text(encoding='utf-8'))
    baseline = baseline_record['metrics']
    baseline_threshold = float(baseline_record['threshold'])
    parent_config = json.loads(parent_config_path.read_text(encoding='utf-8'))
    if abs(float(parent_config['null_threshold']) - baseline_threshold) > 1e-8:
        raise RuntimeError('El umbral original V2 cambio desde el entrenamiento. Revisar antes de comparar.')
    print('Recuperando:', adapter)
    print('No se ejecuta entrenamiento ni se actualiza el adaptador guardado.')
    print('Versiones:', json.dumps(dict(torch=str(torch.__version__),
        transformers=transformers.__version__, peft=peft.__version__)))
    tokenizer = AutoTokenizer.from_pretrained(parent, use_fast=True, local_files_only=True)
    #Carga nueva: no conserva envoltorios de autocast/Accelerate de Trainer.
    base = AutoModelForQuestionAnswering.from_pretrained(parent, local_files_only=True)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False,
        local_files_only=True, autocast_adapter_dtype=True)
    model = model.to(device='cpu', dtype=torch.float32)
    model.eval()
    model.gradient_checkpointing_disable()
    probe = tokenizer(
        ['What color is the sky?', 'Who discovered the comet?', 'What is the last word?'],
        ['The sky is blue.', 'Maria discovered the comet in 2024.',
         ' '.join(['A short sentence about stars.'] * 75) + ' The last word is galaxy.'],
        padding=True, truncation='only_second', max_length=384, return_tensors='pt')
    print('Comparando antes y despues de fusionar: CPU, float32, tres entradas...')
    with torch.inference_mode(), torch.autocast(device_type='cpu', enabled=False):
        before = model(**probe)
        expected = (before.start_logits.detach().clone(), before.end_logits.detach().clone())
        merged = model.merge_and_unload(safe_merge=True)
        merged.eval()
        after = merged(**probe)
        comparison = verify_logits(expected, (after.start_logits, after.end_logits))
    comparison.update(device='cpu', dtype='float32', fresh_load=True, probe_count=3)
    save('comprobacion_fusion.json', comparison)
    for i, item in enumerate(comparison['tensors']):
        print('Inicio' if i == 0 else 'Final', ':', item)
    if not comparison['passed']:
        raise RuntimeError('La comparacion en precision completa no paso. Adaptador intacto; compartir comprobacion_fusion.json. No reentrenar.')
    merged.save_pretrained(output / 'candidato')
    tokenizer.save_pretrained(output / 'candidato')
    del model, base, merged, before, after, expected
    gc.collect()
    #Verificamos la exportacion real cargandola de nuevo para la evaluacion QA.
    candidate_model = AutoModelForQuestionAnswering.from_pretrained(
        output / 'candidato', local_files_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    candidate_model.to(device=device, dtype=torch.float32)
    candidate_model.eval()
    data = load_from_disk(str(root / 'data/squad_v2'))
    examples = list(data['validation'].shuffle(seed=42).select(range(2000)))
    if not {e['id'] for e in examples[:1000]}.isdisjoint(e['id'] for e in examples[1000:]):
        raise RuntimeError('Las particiones de desarrollo se solapan.')
    partition_path = logs / 'particiones.json'
    if partition_path.is_file():
        partitions = json.loads(partition_path.read_text(encoding='utf-8'))
        if (set(partitions['calibration_ids']) != {e['id'] for e in examples[:1000]} or
            set(partitions['monitoring_ids']) != {e['id'] for e in examples[1000:]}):
            raise RuntimeError('Las preguntas no coinciden con las del entrenamiento.')
    print('Evaluando el candidato exportado en precision completa:', device)
    rows = infer_rows(candidate_model, tokenizer, examples)
    threshold = select_threshold(rows[:1000])
    candidate = measure(rows[1000:], threshold)
    save('predicciones_desarrollo_candidato.json', rows)

    def configuration(value, path, source):
        return dict(model_path=str(path.relative_to(root)), source=source,
            null_threshold=float(value), null_rule='return null if difference > null_threshold',
            difference='minimum CLS score minus best span score', max_seq_length=384,
            doc_stride=128, max_answer_length=30, top_k=20,
            threshold_calibration_count=1000, monitoring_count=1000)

    save('umbral_candidato.json', configuration(threshold, output / 'candidato', 'v4_lora_candidate'))
    improved = should_promote(candidate, baseline)
    if improved:
        candidate_model.save_pretrained(output / 'modelo')
        chosen_threshold, selected, chosen = threshold, 'v4_lora_candidate', candidate
    else:
        original = AutoModelForQuestionAnswering.from_pretrained(parent, local_files_only=True)
        original.save_pretrained(output / 'modelo')
        del original
        chosen_threshold, selected, chosen = baseline_threshold, 'parent_v2', baseline
    tokenizer.save_pretrained(output / 'modelo')
    save('umbral.json', configuration(chosen_threshold, output / 'modelo', selected))
    state_path = output / 'trainer_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
    save('resumen.json', dict(selected=selected, baseline_v2=baseline, candidate=candidate,
        selected_metrics=chosen, selected_threshold=chosen_threshold, recovered_from_saved_adapter=True,
        best_candidate_checkpoint=state.get('best_model_checkpoint'),
        global_step=state.get('global_step'), merge_check=comparison,
        note='Desarrollo reutilizado; no es test independiente. API sin modificar.'))
    print('\nV4 FINALIZADA SIN REENTRENAR.')
    print('Seleccion:', selected)
    print('V2:', json.dumps(baseline))
    print('V4 candidato:', json.dumps(candidate))
    print('Modelo seleccionado:', output / 'modelo')
    print('Resumen:', logs / 'resumen.json')


if __name__ == '__main__':
    finalize()
