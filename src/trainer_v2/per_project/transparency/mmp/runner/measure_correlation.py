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


def compute_ranked_list_correlation_put_zero(l1, l2, correlation_fn):
    common_qids = set(l1.keys()).intersection(l2.keys())
    corr_val_list = []
    num_set0 = 0
    for qid in common_qids:
        rl1 = l1[qid]
        rl2 = l2[qid]

        score_d1 = {e.doc_id: e.score for e in rl1}
        score_d2 = {e.doc_id: e.score for e in rl2}

        for doc_id in score_d1.keys():
            if doc_id not in score_d2:
                score_d2[doc_id] = 0
                num_set0 += 1

        scores1 = [score_d1[doc_id] for doc_id in score_d1.keys()]
        scores2 = [score_d2[doc_id] for doc_id in score_d1.keys()]
        corr_val = correlation_fn(scores1, scores2)
        corr_val_list.append(corr_val)

    print(f"{num_set0} docs were set to score 0")
    return corr_val_list



def main():
    correlation_fn = pearson_r_wrap
    first_list_path = sys.argv[1]
    second_list_path = sys.argv[2]

    put0 = False

    if len(sys.argv) > 3:
        option = sys.argv[3]
        if option == "put0":
            put0 = True

    l1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(first_list_path)
    l2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(second_list_path)
    print("A:", first_list_path)
    print("B:", second_list_path)
    if not put0:
        print("From A select query that are in B")
        corr_val_list = compute_ranked_list_correlation(l1, l2, correlation_fn)
    else:
        print(f"Use all from A. If doc not in B, it is set 0")
        corr_val_list = compute_ranked_list_correlation_put_zero(l1, l2, correlation_fn)
    avg_corr = average(corr_val_list)
    print(f"{avg_corr} over {len(corr_val_list)} queries")


if __name__ == "__main__":
    main()
