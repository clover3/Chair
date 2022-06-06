import logging
import random
from collections import Counter
from typing import List, Tuple

import numpy as np

from bert_api.segmented_instance.segmented_text import SegmentedText, get_word_level_segmented_text_from_str
from bert_api.task_clients.nli_interface.nli_interface import NLIPredictorSig, NLIInput
from bert_api.task_clients.nli_interface.nli_predictors import get_nli_cache_client
from contradiction.medical_claims.token_tagging.intersection_search.deletion_tools import Subsequence, do_local_search
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import pick1, Averager
from trainer_v2.chair_logging import c_log


def get_seq_deleter(g_val):
    def delete_fn(num_del, items):
        length = len(items)
        num_del = min(num_del, length)

        def sample_len():
            l = 1
            v = random.random()
            while v < g_val and l < length:
                l = l * 2
                v = random.random()
            return min(l, length)

        delete_indices = []
        for i in range(num_del):
            del_len = sample_len()
            start_idx = pick1(range(length+1))
            end_idx = min(start_idx+del_len, length+1)
            for idx in range(start_idx, end_idx):
                delete_indices.append(idx)
        return delete_indices
    return delete_fn


def get_candidates_next_to_deletions(sub_sequence):
    candidates = []
    for i in sub_sequence.parent.enum_seg_idx():
        if i in sub_sequence.parent_drop_indices:
            pass
        else:
            if i == 0 or i + 1 == sub_sequence.parent.get_seg_len():
                candidates.append(i)
            else:
                if i + 1 in Subsequence.parent_drop_indices or i - 1 in Subsequence.parent_drop_indices:
                    candidates.append(i)
    return candidates


def drop_one_more(sub_sequence: Subsequence) -> Subsequence:
    candidates = get_candidates_next_to_deletions(sub_sequence)
    new_index = pick1(candidates)

    def in_new_indices(i):
        return i in sub_sequence.parent_drop_indices or i == new_index

    new_drop_indices = [i for i in sub_sequence.parent.enum_seg_idx() if in_new_indices(i)]
    return Subsequence(sub_sequence.parent.get_dropped_text(new_drop_indices),
                       sub_sequence.parent,
                       new_drop_indices,
                       )


def drop_sequence(seq_deleter, sub_sequence: Subsequence) -> Subsequence:
    parent = sub_sequence.parent
    all_indices = list(parent.enum_seg_idx())

    remaining_indices = [i for i in all_indices if i not in sub_sequence.parent_drop_indices]
    additional_delete_indices = seq_deleter(4, remaining_indices)

    def in_new_indices(i):
        return i in sub_sequence.parent_drop_indices or i in additional_delete_indices

    new_drop_indices = [i for i in all_indices if in_new_indices(i)]
    return Subsequence(parent.get_dropped_text(new_drop_indices),
                       parent,
                       new_drop_indices,
                       )


def is_left_lower(s1, s2):
    return s1 < s2


Logits = np.array


def get_score_from_records(records: List[Tuple[Subsequence, Logits]]) -> List[float]:
    target_label = 1
    sub, _ = records[0]
    all_indices = list(sub.parent.enum_seg_idx())
    my_avg: List[Averager] = [Averager() for _ in all_indices]
    for subsequence, probs in records:
        for i in all_indices:
            if i not in subsequence.parent_drop_indices:
                my_avg[i].append(probs[target_label])

    scores: List[float] = [a.get_average() for a in my_avg]
    return scores


def search_many_perturbation(predict: NLIPredictorSig, t1, t2) -> List[Tuple[Subsequence, Logits]]:
    records: List[Tuple[Subsequence, Logits]] = []
    # Goal: Modify t2 to get entailed information
    #
    # Run multiple local search, average the scores
    #  1. Delete until neutral drops

    def get_neutral_s(sub_sequence: Subsequence) -> float:
        seg_text = sub_sequence.segmented_text
        X = [NLIInput(t1, seg_text)]
        probs_list = predict(X)
        probs = probs_list[0]
        records.append((sub_sequence, probs))
        return probs[1]

    seq_deleter = get_seq_deleter(0.7)
    init = Subsequence(t2, t2, [])

    def not_neutral(n_trial, current_point: Subsequence, score):
        if score < 0.5:
            return True
        n_tokens = current_point.segmented_text.get_seg_len()
        if n_trial >= n_tokens:
            c_log.debug("terminate after {} trial".format(n_trial))
            return True
        return False

    def drop_sequence_local(seq):
        return drop_sequence(seq_deleter, seq)

    # Do local search by only deleting, goal is to find one with lowest neutral score.

    def do_one_search_run():
        maybe_neutral, info = do_local_search(init, get_neutral_s, drop_sequence_local, is_left_lower, not_neutral)
        c_log.debug(info)
        c_log.debug("got neutral={}".format(get_neutral_s(maybe_neutral)))

    for i in range(5):
        do_one_search_run()

    count_arr: List = summarize_labels(records)
    c_log.info(count_arr)
    # if all neutral
    while min(count_arr) == 0 and len(records) < 200:
        c_log.info("Retry...")
        do_one_search_run()
        count_arr: List = summarize_labels(records)
    c_log.info(count_arr)
    return records


def get_scores_by_many_perturbations(predict_fn, t_text1, t_text2):
    records = search_many_perturbation(predict_fn, t_text1, t_text2)
    scores = get_score_from_records(records)
    return scores


def summarize_labels(records) -> List:
    label_distrib = Counter()
    for _, probs in records:
        label = np.argmax(probs)
        label_distrib[label] += 1
    count_arr: List = [label_distrib[i] for i in range(3)]
    return count_arr


def main():
    problems: List[AlamriProblem] = load_alamri_problem()
    tokenizer = get_tokenizer()
    cache_client = get_nli_cache_client("localhost")
    c_log.setLevel(logging.INFO)

    for p in problems:
        c_log.debug('----')
        t_text1: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, p.text1)
        t_text2: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, p.text2)
        records = search_many_perturbation(cache_client.predict, t_text1, t_text2)
        scores = get_score_from_records(records)



if __name__ == "__main__":
    main()
