from collections import Counter

from cpath import output_path
from misc_lib import path_join, pause_hook
from typing import List, Iterable, Callable, Dict, Tuple, Set
from typing import List, Iterable, Callable, Dict, Tuple, Set

from pytrec_eval import RelevanceEvaluator

from cpath import output_path, at_output_dir
from dataset_specific.msmarco.passage.runner.build_ranked_list import build_ranked_list_from_qid_pid_scores
from misc_lib import path_join, average, TELI

from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_sub_samples
from table_lib import tsv_iter
from misc_lib import select_first_second
from runnable.trec.pytrec_eval_wrap import convert_ranked_list
from trainer_v2.chair_logging import c_log
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def load_ranked_list(run_name):
    dataset = "train_when_0"
    ranked_list_path = path_join(output_path, "ranked_list", f"{run_name}_{dataset}.txt")
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    return ranked_list


def convert_rl_format(entries):
    per_q = {}
    for e in entries:
        per_q[e.doc_id] = e.score
    return per_q


def change_sign(s1, s2):
    diff = s2 - s1
    if diff > 0.01:
        return "+"
    elif diff < -0.01:
        return "-"
    else:
        return "="


def compare_two_runs_inner(qrels, rlg1, rlg2):
    counter = Counter()
    for key in rlg1:
        rel_d = ""
        for k, v in qrels[key].items():
            if v > 0:
                rel_d = k
        if not rel_d:
            continue

        rl_1 = rlg1[key]
        rl_2 = rlg2[key]

        def get_doc_to_rank(rl: List[TrecRankedListEntry]):
            d_t_r = {}
            for i, e in enumerate(rl):
                d_t_r[e.doc_id] = i
            return d_t_r

        doc_id_to_rank1 = get_doc_to_rank(rl_1)
        doc_id_to_rank2 = get_doc_to_rank(rl_2)

        rel_rank1 = doc_id_to_rank1[rel_d] if rel_d in doc_id_to_rank1 else -1
        rel_rank2 = doc_id_to_rank2[rel_d] if rel_d in doc_id_to_rank2 else -1

        if rel_rank1 < 0 or rel_rank2 < 0:
            if rel_rank1 < 0 and rel_rank2 < 0:
                continue
            else:
                print("Rank {} and {} is not expected".format(rel_rank1, rel_rank2))

        reci_r1 = 1 / (rel_rank1 + 1)
        reci_r2 = 1 / (rel_rank2 + 1)

        sign = change_sign(reci_r1, reci_r2)
        counter[sign] += 1
        if sign != "=":
            print(f"Relevant doc : {rel_rank1} -> {rel_rank2} ({reci_r2 - reci_r1})")

    print(counter)

def main():
    run1 = "bm25"
    run2 = "year"

    judgment_path = path_join("data", "msmarco", "qrels.train.tsv")
    qrels = load_qrels_structured(judgment_path)

    rlg1 = load_ranked_list(run1)
    rlg2 = load_ranked_list(run2)
    compare_two_runs_inner(qrels, rlg1, rlg2)


if __name__ == "__main__":
    main()