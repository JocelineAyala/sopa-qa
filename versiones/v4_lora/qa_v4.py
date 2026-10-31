"""Postprocesamiento y métricas de QA; no carga modelos ni usa APIs externas."""

# Permite ejecutar este archivo directamente desde cualquier directorio.
import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import ROOT


import math
import re
import string
from collections import Counter

import numpy as np


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score_answer(text, references):
    references = [r for r in references if normalize(r)] or ['']
    prediction = normalize(text).split()
    best_em = best_f1 = 0.0
    for reference in references:
        gold = normalize(reference).split()
        em = float(prediction == gold)
        if not prediction or not gold:
            f1 = em
        else:
            common = sum((Counter(prediction) & Counter(gold)).values())
            f1 = 2 * common / (len(prediction) + len(gold))
        best_em, best_f1 = max(best_em, em), max(best_f1, f1)
    return best_em, best_f1


def decode(examples, metadata, start_logits, end_logits, max_answer_length=30):
    #Comparamos los fragmentos de todas las ventanas de cada pregunta.
    if len(metadata) != len(start_logits) or len(metadata) != len(end_logits):
        raise ValueError('El número de ventanas y predicciones no coincide.')
    best = [None] * len(examples)
    null_scores = [math.inf] * len(examples)
    for feature, start, end in zip(metadata, start_logits, end_logits):
        index = feature['example_index']
        cls = feature['cls_index']
        null = float(start[cls] + end[cls])
        null_scores[index] = min(null_scores[index], null)
        offsets = feature['offsets']
        positions = np.array([i for i, offset in enumerate(offsets)
                              if offset is not None and offset[1] > offset[0]])
        if not len(positions):
            continue
        starts = positions[np.argsort(-np.asarray(start)[positions], kind='stable')[:20]]
        ends = positions[np.argsort(-np.asarray(end)[positions], kind='stable')[:20]]
        for s in starts:
            for e in ends:
                if e < s or e - s + 1 > max_answer_length:
                    continue
                left, right = offsets[s][0], offsets[e][1]
                value = float(start[s] + end[e])
                if right > left and (best[index] is None or value > best[index][0]):
                    best[index] = (value, left, right)
    rows = []
    for index, example in enumerate(examples):
        if best[index] is None:
            raise RuntimeError(f"Sin candidato: {example['id']}")
        value, left, right = best[index]
        text = example['context'][left:right]
        difference = null_scores[index] - value
        if not math.isfinite(difference):
            raise RuntimeError('Puntuación no finita.')
        references = example['answers']['text']
        em_span, f1_span = score_answer(text, references)
        em_null, f1_null = score_answer('', references)
        rows.append(dict(id=example['id'], span=text, start=left,
                         difference=difference, has_answer=bool(references),
                         references=references, em_span=em_span, f1_span=f1_span,
                         em_null=em_null, f1_null=f1_null))
    return rows


def harmonic(a, b):
    return 2 * a * b / (a + b) if a + b else 0.0


def should_promote(candidate, baseline):
    return (candidate['harmonic_EM'] > baseline['harmonic_EM'] + 1e-8
            and candidate['answerable']['EM'] >= baseline['answerable']['EM'] - 1e-8
            and candidate['unanswerable']['EM'] >= baseline['unanswerable']['EM'] - 1e-8)


def measure(rows, threshold):
    def group(items):
        if not items:
            raise ValueError('Cada partición necesita ambos tipos de pregunta.')
        em = f1 = 0.0
        for row in items:
            suffix = 'null' if row['difference'] > threshold else 'span'
            em += row['em_' + suffix]
            f1 += row['f1_' + suffix]
        return dict(count=len(items), EM=100 * em / len(items), F1=100 * f1 / len(items))
    result = dict(overall=group(rows),
                  answerable=group([r for r in rows if r['has_answer']]),
                  unanswerable=group([r for r in rows if not r['has_answer']]))
    result['harmonic_EM'] = harmonic(result['answerable']['EM'], result['unanswerable']['EM'])
    result['balanced_EM'] = (result['answerable']['EM'] + result['unanswerable']['EM']) / 2
    return result


def select_threshold(rows):
    #Maximizamos la media armónica de los aciertos de ambas clases.
    #Solo se usan las filas de calibración, nunca las de seguimiento.
    ordered = sorted(rows, key=lambda r: r['difference'])
    if not ordered:
        raise ValueError('Calibración vacía.')
    na = sum(r['has_answer'] for r in rows)
    nn = len(rows) - na
    if not na or not nn:
        raise ValueError('Calibración necesita preguntas con y sin respuesta.')
    em_a = sum(r['em_null'] for r in rows if r['has_answer'])
    em_n = sum(r['em_null'] for r in rows if not r['has_answer'])
    f1_a = sum(r['f1_null'] for r in rows if r['has_answer'])
    f1_n = sum(r['f1_null'] for r in rows if not r['has_answer'])
    def quality():
        a, n = em_a / na, em_n / nn
        return (round(harmonic(a, n), 12),
                round((a + n) / 2, 12),
                round((f1_a / na + f1_n / nn) / 2, 12))
    threshold = math.nextafter(ordered[0]['difference'], -math.inf)
    best_threshold, best_quality = threshold, quality()
    i = 0
    while i < len(ordered):
        threshold = ordered[i]['difference']
        while i < len(ordered) and ordered[i]['difference'] == threshold:
            row = ordered[i]
            delta_em = row['em_span'] - row['em_null']
            delta_f1 = row['f1_span'] - row['f1_null']
            if row['has_answer']:
                em_a += delta_em
                f1_a += delta_f1
            else:
                em_n += delta_em
                f1_n += delta_f1
            i += 1
        candidate = quality()
        if candidate > best_quality:
            best_threshold, best_quality = threshold, candidate
    return best_threshold


def tokenize_examples(examples, tokenizer, max_length=384, stride=128):
    #Estas características incluyen etiquetas para calcular la pérdida.
    from datasets import Dataset
    records, metadata = [], []
    for index, example in enumerate(examples):
        encoded = tokenizer(example['question'].strip(), example['context'],
                            max_length=max_length, stride=stride,
                            truncation='only_second', padding='max_length',
                            return_overflowing_tokens=True, return_offsets_mapping=True)
        for i, ids in enumerate(encoded['input_ids']):
            cls = ids.index(tokenizer.cls_token_id)
            sequence = encoded.sequence_ids(i)
            offsets = [tuple(v) if sequence[j] == 1 else None
                       for j, v in enumerate(encoded['offset_mapping'][i])]
            positions = [j for j, v in enumerate(offsets) if v is not None and v[1] > v[0]]
            start = end = cls
            answers = example['answers']
            if answers['text'] and positions:
                left = answers['answer_start'][0]
                right = left + len(answers['text'][0])
                first, last = positions[0], positions[-1]
                if offsets[first][0] <= left and offsets[last][1] >= right:
                    start = first
                    while start <= last and offsets[start][0] <= left:
                        start += 1
                    start -= 1
                    end = last
                    while end >= first and offsets[end][1] >= right:
                        end -= 1
                    end += 1
            record = {key: encoded[key][i] for key in tokenizer.model_input_names if key in encoded}
            record.update(start_positions=start, end_positions=end)
            records.append(record)
            metadata.append(dict(example_index=index, cls_index=cls, offsets=offsets))
    return Dataset.from_list(records), metadata


def infer_rows(model, tokenizer, examples, batch_size=2):
    import torch
    from torch.utils.data import DataLoader
    from transformers import default_data_collator
    from tqdm.auto import tqdm
    features, metadata = tokenize_examples(examples, tokenizer)
    loader = DataLoader(features, batch_size=batch_size, collate_fn=default_data_collator)
    device = next(model.parameters()).device
    starts, ends = [], []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc='Predicciones QA'):
                inputs = {k: v.to(device) for k, v in batch.items()
                          if k not in ('start_positions', 'end_positions')}
                output = model(**inputs)
                starts.append(output.start_logits.float().cpu().numpy())
                ends.append(output.end_logits.float().cpu().numpy())
    finally:
        model.train(was_training)
    return decode(examples, metadata, np.concatenate(starts), np.concatenate(ends))


def prepare_training_batch(ejemplos, tokenizer):
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
