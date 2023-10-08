import sys
from typing import List, Iterable, Dict

from list_lib import flatten
from misc_lib import average
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import pearson_r_wrap
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def compute_ranked_list_correlation(l1, l2, correlation_fn):
    common_qids = set(l1.keys()).intersection(l2.keys())
    corr_val_list = []
    for qid in common_qids:
        rl1 = l1[qid]
        rl2 = l2[qid]

        score_d1 = {e.doc_id: e.score for e in rl1}
        score_d2 = {e.doc_id: e.score for e in rl2}
        common_doc_ids = set(score_d1.keys()).intersection(score_d2.keys())
        if len(common_doc_ids) != len(score_d1) or \
                len(common_doc_ids) != len(score_d2):
            pass
            print(f"Query {qid} has only {len(common_doc_ids)} in common")

        scores1 = [score_d1[doc_id] for doc_id in common_doc_ids]
        scores2 = [score_d2[doc_id] for doc_id in common_doc_ids]
        corr_val = correlation_fn(scores1, scores2)
        corr_val_list.append(corr_val)
    return corr_val_list


def main():
    correlation_fn = pearson_r_wrap
    first_list_path = sys.argv[1]
    second_list_path = sys.argv[2]
    print("From {} select query that are in {}".format(first_list_path, second_list_path))
    l1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(first_list_path)
    l2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(second_list_path)
    corr_val_list = compute_ranked_list_correlation(l1, l2, correlation_fn)
    avg_corr = average(corr_val_list)
    print(f"{avg_corr} over {len(corr_val_list)} queries")



if __name__ == "__main__":
    main()