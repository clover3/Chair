import pickle
import sys
from collections import Counter
from typing import List, Iterable, Tuple

from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.passage_resource_loader import FourItem, tsv_iter, enum_grouped
from dataset_specific.msmarco.passage.path_helper import get_mmp_train_grouped_sorted_path

# Output (Doc_id, TFs Counter, base score)
from misc_lib import TELI
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import tfs_and_scores_path_train


def precompute_ranked_list(
        itr: Iterable[List[FourItem]], bm25: BM25, n_item) -> Iterable[List[Tuple[str, Counter, float]]]:
    for group in TELI(itr, n_item):
        per_query_entries = []
        for qid, pid, query, text in group:
            q_terms = bm25.tokenizer.tokenize_stem(query)
            t_terms = bm25.tokenizer.tokenize_stem(text)
            q_tf = Counter(q_terms)
            t_tf = Counter(t_terms)
            score = bm25.score_inner(q_tf, t_tf)
            per_query_entries.append((pid, t_tf, score))
        yield per_query_entries


def compute_base_score(itr, job_no, get_tfs_scores_save_path, n_item):
    g_itr: Iterable[List[FourItem]] = enum_grouped(itr)
    bm25 = get_bm25_mmp_25_01_01()
    step_size = 100
    out_itr = precompute_ranked_list(g_itr, bm25, n_item)
    sub_no = 0
    output = []

    def flush():
        nonlocal output, sub_no
        if not output:
            return
        save_name = f"{job_no}_{sub_no}"
        save_path = get_tfs_scores_save_path(save_name)
        pickle.dump(output, open(save_path, "wb"))
        output = []
        sub_no += 1

    for g in out_itr:
        output.append(g)
        if len(output) > step_size:
            flush()
            c_log.info("Wrote {} groups".format(step_size))
    flush()
    c_log.info("Done")


def main():
    job_no = sys.argv[1]
    src_path = get_mmp_train_grouped_sorted_path(job_no)
    itr = tsv_iter(src_path)
    get_tfs_scores_save_path = tfs_and_scores_path_train
    n_item = 3000
    compute_base_score(itr, job_no, get_tfs_scores_save_path, n_item)



if __name__ == "__main__":
    main()