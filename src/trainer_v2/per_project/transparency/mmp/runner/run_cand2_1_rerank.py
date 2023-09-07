from collections import defaultdict

from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25t import BM25T
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_and_save_scores, eval_dev_mrr
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper
from typing import Dict


def load_mapping_from_align_scores(
        tsv_path, cut, mapping_val) -> Dict[str, Dict[str, float]]:
    rows = tsv_iter(tsv_path)

    n_entry = 0
    mapping = defaultdict(dict)
    for q_term, d_term, score in rows:
        if float(score) > cut:
            mapping[q_term][d_term] = mapping_val
            n_entry += 1

    c_log.info("%d entry loaded", n_entry)
    return mapping


def run_eval_with_bm25t(dataset, mapping, run_name):
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, 0.1, 100, 1.4)
    bm25t = BM25T(mapping, bm25.core)
    predict_and_save_scores(bm25t.score, dataset, run_name, 1000 * 1000)
    score = eval_dev_mrr(dataset, run_name)
    print(f"mrr:\t{score}")


def cand_2_1():
    ph = get_cand2_1_path_helper()
    table_path = ph.per_pair_candidates.fidelity_table_path
    dataset = "dev_sample1000"
    cut = 0.1
    mapping_val = 0.1
    table_name = f"cand2_1_cut{cut}"
    run_name = f"bm25_{table_name}"
    mapping = load_mapping_from_align_scores(table_path, cut, mapping_val)
    run_eval_with_bm25t(dataset, mapping, run_name)


def main_1000_eval():
    dataset = "dev_sample1000"
    table_name = "cand2_1"
    run_name = f"bm25_{table_name}"
    score = eval_dev_mrr(dataset, run_name)
    print(f"mrr:\t{score}")


def main():
    cand_2_1()


if __name__ == "__main__":
    main()
