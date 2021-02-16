from typing import List, Iterable, Dict

import numpy as np

from attribution.baselines import get_real_len, informative_fn_eq1, int_list_to_str
from data_generator.subword_convertor import SubwordConvertor

MASK_ID = 103


def replace_tokens(binary_tag, input_ids, input_mask, segment_ids):
    assert len(input_ids) == len(input_mask)
    assert len(input_ids) == len(segment_ids)
    assert len(input_ids) == len(binary_tag)

    length = len(binary_tag)
    x0_new = []
    x1_new = []
    x2_new = []

    for i in range(length):
        if binary_tag[i]:
            x0_new.append(MASK_ID)
            x1_new.append(input_mask[i])
            x2_new.append(segment_ids[i])
        else:
            x0_new.append(input_ids[i])
            x1_new.append(input_mask[i])
            x2_new.append(segment_ids[i])
    while len(x0_new) < len(input_ids):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    return x0_new, x1_new, x2_new


def get_location_to_delete(real_len, target_term, input_ids, segment_ids):
    n_match = 0
    location_to_del = []
    for i in range(real_len):
        try:
            if input_ids[i] == target_term[n_match] and segment_ids[i]:
                n_match += 1
                if n_match == len(target_term):
                    ed = i + 1
                    for j in range(ed - n_match, ed):
                        location_to_del.append(j)
                    n_match = 0
            else:
                n_match = 0
        except IndexError:
            print('target_term', target_term)
            print('i', i)
            print('n_match', n_match)
            print('input_ids', input_ids)
            raise

    return location_to_del


def remove_term(target_term: List[int], input_ids, input_mask, segment_ids, real_len):
    assert len(input_ids) == len(input_mask)
    assert len(input_ids) == len(segment_ids)

    location_to_del = get_location_to_delete(real_len, target_term, input_ids, segment_ids)

    x0_new = []
    x1_new = []
    x2_new = []

    for i in range(real_len):
        if i in location_to_del and segment_ids[i]:
            pass
        else:
            x0_new.append(input_ids[i])
            x1_new.append(input_mask[i])
            x2_new.append(segment_ids[i])

    while len(x0_new) < len(input_ids):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    return x0_new, x1_new, x2_new


def replace_term(target_term, input_ids, input_mask, segment_ids, real_len):
    assert len(input_ids) == len(input_mask)
    assert len(input_ids) == len(segment_ids)

    x0_new = []
    x1_new = []
    x2_new = []
    location_to_del = get_location_to_delete(real_len, target_term, input_ids, segment_ids)

    for i in range(real_len):
        if i in location_to_del:
            x0_new.append(MASK_ID)
            x1_new.append(input_mask[i])
            x2_new.append(segment_ids[i])
        else:
            x0_new.append(input_ids[i])
            x1_new.append(input_mask[i])
            x2_new.append(segment_ids[i])

    while len(x0_new) < len(input_ids):
        x0_new.append(0)
        x1_new.append(0)
        x2_new.append(0)
    return x0_new, x1_new, x2_new


def explain_by_replace(data, target_tag, forward_run) -> List[np.array]:
    inputs = []
    inputs_info = []
    base_indice = []
    for entry in data:
        input_ids, input_mask, segment_ids = entry
        base_case = entry
        base_case_idx = len(inputs_info)
        base_indice.append(base_case_idx)
        inputs.append(base_case)

        seq_len = len(input_ids)
        real_len = get_real_len(input_mask, seq_len)
        info = {
            'base_case_idx': base_case_idx,
            'type': 'base_run',
            'seq_len': seq_len,
        }
        inputs_info.append(info)

        for idx in range(real_len):
            mask = np.zeros([seq_len])
            mask[idx] = 1
            new_case = replace_tokens(mask, input_ids, input_mask, segment_ids)
            info = {
                'base_case_idx': base_case_idx,
                'type': 'mod',
                'mod_idx': idx
            }
            inputs.append(new_case)
            inputs_info.append(info)

    logits_list = forward_run(inputs)
    logit_attrib_list = informative_fn_eq1(target_tag, logits_list)
    assert len(logits_list) == len(inputs)

    explains = []
    for case_idx in base_indice:
        base_ce = logit_attrib_list[case_idx]
        idx = case_idx + 1
        seq_len = inputs_info[case_idx]['seq_len']
        attrib_scores = np.zeros([seq_len]) - 100

        mod_set = set()
        while idx < len(inputs_info) and inputs_info[idx]['base_case_idx'] == case_idx:
            after_ce = logit_attrib_list[idx]
            diff_ce = base_ce - after_ce
            score = diff_ce
            mod_idx = inputs_info[idx]['mod_idx']
            assert mod_idx not in mod_set
            mod_set.add(mod_idx)
            attrib_scores[mod_idx] = score
            idx += 1

        explains.append(attrib_scores)
    return explains


def explain_by_term_deletion(data, target_tag, forward_run) -> List[Dict]:
    return explain_by_term_level_op(data, target_tag, forward_run, remove_term)


def explain_by_term_replace(data, target_tag, forward_run) -> List[Dict]:
    return explain_by_term_level_op(data, target_tag, forward_run, replace_term)


def explain_by_term_level_op(data, target_tag, forward_run, perturb_op) -> List[Dict]:
    sb_word = SubwordConvertor()
    inputs = []
    inputs_info = []
    base_indice = []
    for entry in data:
        x0, x1, x2 = entry
        base_case = entry
        base_case_idx = len(inputs_info)
        base_indice.append(base_case_idx)
        inputs.append(base_case)

        seq_len = len(x0)
        real_len = get_real_len(x1, seq_len)
        info = {
            'base_case_idx': base_case_idx,
            'type': 'base_run',
            'seq_len': seq_len,
        }
        inputs_info.append(info)

        input_ids = x0[:real_len]
        unique_terms: Iterable[List[int]] = sb_word.get_word_as_subtokens(input_ids)

        for term in unique_terms:
            if not term:
                continue
            new_case = perturb_op(term, x0, x1, x2, real_len)
            info = {
                'base_case_idx': base_case_idx,
                'type': 'mod',
                'mod_term': term
            }
            inputs.append(new_case)
            inputs_info.append(info)

    logits_list = forward_run(inputs)
    logit_attrib_list = informative_fn_eq1(target_tag, logits_list)
    assert len(logits_list) == len(inputs)

    explains = []
    for case_idx in base_indice:
        base_ce = logit_attrib_list[case_idx]
        idx = case_idx + 1
        attrib_scores = {}

        diff_max = -1
        while idx < len(inputs_info) and inputs_info[idx]['base_case_idx'] == case_idx:
            after_ce = logit_attrib_list[idx]
            diff_ce = base_ce - after_ce
            diff_max = max(diff_ce, diff_max)
            score = diff_ce
            term = inputs_info[idx]['mod_term']
            attrib_scores[int_list_to_str(term)] = score
            idx += 1

        explains.append(attrib_scores)
    return explains


