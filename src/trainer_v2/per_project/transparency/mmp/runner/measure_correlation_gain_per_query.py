import sys
from typing import List, Iterable, Dict

from list_lib import flatten, left, right
from misc_lib import average
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import pearson_r_wrap, spearman_r_wrap
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def compute_ranked_list_correlation(
        l_ref, l_base, l_target, correlation_fn):
    corr_val_list = []
    for qid in l_ref:
        ref_rl = l_ref[qid]
        score_d_ref = {e.doc_id: e.score for e in ref_rl}

        try:
            rl1 = l_base[qid]
            rl2 = l_target[qid]

            score_d1 = {e.doc_id: e.score for e in rl1}
            score_d2 = {e.doc_id: e.score for e in rl2}
            common_doc_ids = set(score_d1.keys())\
                .intersection(score_d2.keys())\
                .intersection(score_d_ref.keys())

            if len(common_doc_ids) != len(score_d1) or \
                    len(common_doc_ids) != len(score_d2):
                print(f"Query {qid} has only {len(common_doc_ids)} in common")

            scores_ref = [score_d_ref[doc_id] for doc_id in common_doc_ids]
            scores1 = [score_d1[doc_id] for doc_id in common_doc_ids]
            scores2 = [score_d2[doc_id] for doc_id in common_doc_ids]
            corr_val1 = correlation_fn(scores_ref, scores1)
            corr_val2 = correlation_fn(scores_ref, scores2)
            corr_val_list.append((corr_val1, corr_val2))
        except KeyError:
            pass
    return corr_val_list


def main():
    correlation_fn = pearson_r_wrap
    ref_path = sys.argv[1]
    base_path = sys.argv[2]
    target_path = sys.argv[3]

    l_ref: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ref_path)
    l_base: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(base_path)
    l_target: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(target_path)
    corr_val_list = compute_ranked_list_correlation(l_ref, l_base, l_target, correlation_fn)

    for v1, v2 in corr_val_list:
        print(f"{v2-v1:.4f}\t{v1}")

    gain = average(left(corr_val_list)) - average(right(corr_val_list))
    print(f"{gain} over {len(corr_val_list)} queries")


if __name__ == "__main__":
    main()