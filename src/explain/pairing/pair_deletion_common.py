import csv
import os
from collections import defaultdict
from typing import NamedTuple, List, Any, Tuple, Dict

import scipy.special
from numpy.core.multiarray import ndarray

from cpath import data_path
from data_generator.tokenizer_wo_tf import EncoderUnitPlain
from trainer.np_modules import get_batches_ex


class Deletion(NamedTuple):
    text: str
    del_loc: int


def single_deletion_gen(tokens) -> List[Deletion]:
    output = [Deletion(" ".join(tokens), -1)]
    for i in range(len(tokens)):
        new_tokens = tokens[:i] + tokens[i+1:]
        d = Deletion(" ".join(new_tokens), i)
        output.append(d)
    return output


def deletion_gen_iter(p, h):
    p_tokens = p.split()
    h_tokens = h.split()

    for p_deletion in single_deletion_gen(p_tokens):
        for h_deletion in single_deletion_gen(h_tokens):
            yield p_deletion, h_deletion


def deletion_gen(text_pair_list, pair_encode_fn):
    info_entries = []
    raw_payload = []
    for idx, (p, h) in enumerate(text_pair_list):
        for dp, dh in deletion_gen_iter(p, h):
            info = {
                'data_idx': idx,
                'p_idx': dp.del_loc,
                'h_idx': dh.del_loc,
            }
            x1,x2,x3 = pair_encode_fn(dp.text, dh.text)
            e = x1, x2, x3, 0
            raw_payload.append(e)
            info_entries.append(info)

    return info_entries, raw_payload


def get_payload(input_path, nli_setting, batch_size) -> Tuple[Any, List]:
    initial_text = load_p_h_pair_text(input_path)
    voca_path = os.path.join(data_path, nli_setting.vocab_filename)
    encoder_unit = EncoderUnitPlain(nli_setting.seq_length, voca_path)
    info_entries, raw_payload = deletion_gen(initial_text, encoder_unit.encode_pair)
    batches = get_batches_ex(raw_payload, batch_size, 4)
    return batches, info_entries


def load_p_h_pair_text(input_path):
    f = open(input_path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter=',')
    initial_text = []
    for row in reader:
        s1 = row[0]
        s2 = row[1]
        initial_text.append((s1, s2))
    return initial_text


class PerGroupSummary(NamedTuple):
    group_idx: int
    score_d: Dict[Tuple[int, int], ndarray]
    effect_d: Dict[Tuple[int, int], Tuple[float, float]]


def summarize_pair_deletion_results(info_entries, output_d) -> List[PerGroupSummary]:
    logits_list = output_d['logits']
    target_idx = 1


    grouped = defaultdict(list)
    for info, logits in zip(info_entries, logits_list):
        info['logits'] = logits
        grouped[info['data_idx']].append(info)

    per_group_summary = []
    for group_idx, entries in grouped.items():
        score_d = {}
        for e in entries:
            p_idx = e['p_idx']
            h_idx = e['h_idx']
            score_d[(p_idx, h_idx)] = scipy.special.softmax(e['logits'])

        def get_score_at(key):
            return score_d[key][target_idx]

        base_score = get_score_at((-1, -1))
        effect_d = {}
        for p_idx, h_idx in score_d:
            if p_idx == -1 or h_idx == -1:
                continue
            p_only_score = get_score_at((p_idx, -1))
            h_only_score = get_score_at((-1, h_idx))
            pairwise_score = get_score_at((p_idx, h_idx))

            # if contradiction and this pair is matching ones
            # 0.1   0.3   0.1  0.1
            pair_effect_by_p = p_only_score - pairwise_score
            pair_effect_by_h = h_only_score - pairwise_score
            effect_d[(p_idx, h_idx)] = pair_effect_by_p, pair_effect_by_h

        per_group_summary.append(PerGroupSummary(group_idx, score_d, effect_d))
    return per_group_summary