import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Tuple, List

import numpy as np

from cache import save_to_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.evidence_selector_by_attn import get_merged_attn_scores
from data_generator2.segmented_enc.es_common.pep_attn_common import PairWithAttn
from data_generator2.segmented_enc.es_mmp.pep_attn_common import iter_attention_mmp_pos_neg_paired
from dataset_specific.msmarco.passage.path_helper import train_triples_small_partition_iter
from list_lib import lflatten
from misc_lib import TELI
from taskman_client.wrapper3 import JobContext
from trainer.promise import PromiseKeeper
from trainer_v2.custom_loop.definitions import ModelConfig512_1, ModelConfig256_1
from trainer_v2.custom_loop.neural_network_def.ts_concat_distil import load_part_model_from_ts_concat_distill, \
    TSInference
from trainer_v2.per_project.transparency.mmp.pep.demo_util import PEPLocalDecision
from trainer_v2.per_project.transparency.mmp.pep.runner.run_pep_segment_score_view import mask_remain
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TokenizedText, merge_subword_scores, \
    is_valid_indices, enum_neighbor, get_term_rep, TrialLogger
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def enum_ngram_indices(seq_len, n):
    for i in range(seq_len):
        indices = [i]
        for j in range(i+1, i+n):
            if j < seq_len:
                indices.append(j)
        yield indices


def analyze_inner(pair_itr: Iterable[Tuple[str, str, int]], score_fn):
    @dataclass
    class Span:
        indices: list[int]
        tokens: list[str]
        key: str

    tokenizer = get_tokenizer()
    for e in TELI(pair_itr, 3000):
        q, d, label = e
        q_rep: TokenizedText = TokenizedText.from_text(tokenizer, q)

        d_rep: TokenizedText = TokenizedText.from_text(tokenizer, d)
        d_tokens = lflatten(d_rep.sb_tokens)
        seen = set()
        q_sp_len = len(q_rep.sp_tokens)
        payload = []
        payload_info: List[Span] = []
        n = 1
        for n in [1, 2]:
            for q_indice in enum_ngram_indices(q_sp_len, n):
                ngram_tokens = get_span_rep(q_rep, q_indice)

                key: str = " ".join(ngram_tokens)
                if key not in seen:
                    payload.append([ngram_tokens, d_tokens])
                    payload_info.append(Span(q_indice, ngram_tokens, key))
                    seen.add(key)

        scores = score_fn(payload)
        score_d: dict[str, float] = {}
        # For each bigram, compare with sum of unigram
        for info, score in zip(payload_info, scores):
            score_d[info.key] = score

        for info, score in zip(payload_info, scores):
            try:
                if len(info.indices) == 2:
                    [i1, i2] = info.indices
                    ngram_tokens = get_span_rep(q_rep, [i1])
                    key: str = " ".join(ngram_tokens)
                    score1 = score_d[key]
                    ngram_tokens = get_span_rep(q_rep, [i2])
                    key: str = " ".join(ngram_tokens)
                    score2 = score_d[key]
                    change = score - (score1 + score2)
                    log_entry = [label, info.key, score, change]
                else:
                    log_entry = [label, info.key, score, "N/A"]
                yield log_entry
            except KeyError as e:
                print(e)


def get_span_rep(q_rep, q_indice):
    q_idx_st = q_indice[0]
    q_idx_ed = q_indice[-1]
    ngram_tokens: List[str] = q_rep.get_sb_list_form_sp_indices(q_indice)
    if q_idx_st > 0:
        ngram_tokens = ["[MASK]"] + ngram_tokens
    if q_idx_ed < len(q_rep.sp_tokens):
        ngram_tokens = ngram_tokens + ["[MASK]"]
    return ngram_tokens




def analyze(model_path,
            log_path,
            itr: Iterable[Tuple[str, str]],
            ):
    strategy = get_strategy()
    with strategy.scope():
        model_config = ModelConfig512_1()
        model = load_part_model_from_ts_concat_distill(model_path, model_config)
        helper = TSInference(model_config, model)
    with open(log_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        max_change = -1000
        for log_entry in analyze_inner(itr, helper.score_from_tokenized_fn):
            writer.writerow(log_entry)
            label, key, score, change = log_entry
            if isinstance(change, float) and change > max_change:
                print("New max change: ", key, score, change)
                max_change = change
            file.flush()




def main():
    # Goal: check which span pairs are important in PEP.
    #   Method: For each qd, apply a few pairs, record top scoring one.
    #
    #   How to handle exact match?
    #       - Exact match having high score is trivial. So it may not be worth checking it.
    #       - However, they don't always result in the same score.
    #            Some (rare words) has higher score than other (frequent ones)

    model_path = sys.argv[1]
    log_path = sys.argv[2]
    partition_no = int(sys.argv[3])

    with JobContext(f"TermPair_{partition_no}"):
        # pos_neg_pair_itr = iter_attention_mmp_pos_neg_paired(partition_no)
        raw_train_iter = train_triples_small_partition_iter(partition_no)

        def pos_doc_pair_itr():
            for q, dp, dn in raw_train_iter:
                yield q, dp, 1
                yield q, dn, 1
        analyze(model_path, log_path, pos_doc_pair_itr())


if __name__ == "__main__":
    main()
