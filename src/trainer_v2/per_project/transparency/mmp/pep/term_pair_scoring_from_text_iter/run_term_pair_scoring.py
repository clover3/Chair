import sys
from collections import defaultdict
from typing import Iterable, Tuple, List

import numpy as np

from cache import save_to_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.evidence_selector_by_attn import get_merged_attn_scores
from data_generator2.segmented_enc.es_common.pep_attn_common import PairWithAttn
from data_generator2.segmented_enc.es_mmp.pep_attn_common import iter_attention_mmp_pos_neg_paired
from misc_lib import TELI
from taskman_client.wrapper3 import JobContext
from trainer.promise import PromiseKeeper
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.per_project.transparency.mmp.pep.demo_util import PEPLocalDecision
from trainer_v2.per_project.transparency.mmp.pep.runner.run_pep_segment_score_view import mask_remain
from trainer_v2.per_project.transparency.mmp.pep.seg_enum_helper import TokenizedText, merge_subword_scores, \
    is_valid_indices, enum_neighbor, get_term_rep, TrialLogger
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def analyze_inner(itr, log_path, pep, top_k):
    max_per_pair = 10
    max_seq_length = 256
    full_logger = TrialLogger(log_path)
    score_d = defaultdict(list)
    tokenizer = get_tokenizer()
    for e in TELI(itr, 3000):
        (pos_pair, pos_attn), (neg_pair, neg_attn) = e
        q: str = pos_pair.segment1
        q_rep: TokenizedText = TokenizedText.from_text(tokenizer, q)
        pair = pos_pair
        attn = pos_attn

        d = pair.segment2
        d_rep: TokenizedText = TokenizedText.from_text(tokenizer, d)
        pk = PromiseKeeper(pep.score_fn)

        payload = []

        def add_entry(q_indices, d_indices):
            q_masked = mask_remain(q_indices, q_rep.sb_tokens)
            d_masked = mask_remain(d_indices, d_rep.sb_tokens)
            score_future = pk.get_future((q_masked, d_masked))
            payload.append((q_rep, d_rep, q_indices, d_indices, score_future))

        seen_current_pair = set()

        def add_entry_if_valid(q_indices: List[int], d_indices: List[int]):
            if is_valid_indices(q_rep, q_indices) and is_valid_indices(d_rep, d_indices):
                pass
            else:
                return

            q_term = get_term_rep(q_rep, q_indices)
            d_term = get_term_rep(d_rep, d_indices)
            key = q_term, d_term

            if len(score_d[key]) >= max_per_pair:
                return

            if key in seen_current_pair:
                pass
            else:
                add_entry(q_indices, d_indices)

                seen_current_pair.add(key)

        for q_idx in range(len(q_rep.sp_tokens)):
            part_in_seg_st, part_in_seg_ed = q_rep.get_sb_range(q_idx)

            # Shape [2, len(d_rep.sb_tokens)]
            part_seg_part_i_mean = get_merged_attn_scores(
                attn, q_rep.get_sb_len(), d_rep.get_sb_len(),
                part_in_seg_st, part_in_seg_ed)

            cur_seq_len = q_rep.get_sb_len() + d_rep.get_sb_len() + 3
            score_for_d_rep = part_seg_part_i_mean[1]
            if max_seq_length < cur_seq_len:
                if len(score_for_d_rep) < d_rep.get_sb_len():
                    dummy = [0] * (d_rep.get_sb_len() - len(score_for_d_rep))
                    score_for_d_rep = np.concatenate([score_for_d_rep, dummy])
            scores: List[float] = merge_subword_scores(score_for_d_rep, d_rep, max)
            score_rank = np.argsort(scores)[::-1]
            score_rank = score_rank.tolist()

            top_indices: List[int] = score_rank[:top_k]
            for d_idx in top_indices:
                add_entry_if_valid([q_idx], [d_idx])

                for q_indices in enum_neighbor(q_idx):
                    add_entry_if_valid(q_indices, [d_idx])

                for d_indices in enum_neighbor(d_idx):
                    add_entry_if_valid([q_idx], d_indices)

        pk.do_duty(True)

        full_logger.log_rep(q_rep, d_rep)
        for q_rep, d_rep, q_indices, d_indices, score_future in payload:
            score = float(score_future.get())
            q_term = get_term_rep(q_rep, q_indices)
            d_term = get_term_rep(d_rep, d_indices)
            full_logger.log_score(q_indices, d_indices, score)
            score_d[q_term, d_term].append(score)


def analyze(model_path,
            log_path,
            itr: Iterable[Tuple[PairWithAttn, PairWithAttn]],
            top_k=5):

    strategy = get_strategy()
    with strategy.scope():
        model_config = ModelConfig512_1()
        pep = PEPLocalDecision(model_config, model_path)

    analyze_inner(itr, log_path, pep, top_k)


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
        pos_neg_pair_itr = iter_attention_mmp_pos_neg_paired(partition_no)
        analyze(model_path, log_path, pos_neg_pair_itr)


if __name__ == "__main__":
    main()
