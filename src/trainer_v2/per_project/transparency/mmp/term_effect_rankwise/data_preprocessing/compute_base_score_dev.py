import sys
from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set
from dataset_specific.msmarco.passage.passage_resource_loader import FourItem, tsv_iter, enum_grouped
from dataset_specific.msmarco.passage.path_helper import get_mmp_train_grouped_sorted_path, get_mmp_grouped_sorted_path

# Output (Doc_id, TFs Counter, base score)
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.data_preprocessing.resplit_tfs import save_qid_tfs
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.data_preprocessing.serializer import \
    save_shallow_scores
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_shallow_score_save_path


def compute_base_score(itr, shallow_score_save_path):
    g_itr: Iterable[List[FourItem]] = enum_grouped(itr)
    bm25 = get_bm25_mmp_25_01_01()

    qid_scores: List[Tuple[str, List[Tuple[str, float]]]] = []
    for group in g_itr:
        tfs_list: List[Tuple[str, Counter]] = []
        scores: List[Tuple[str, float]] = []
        qid = group[0][0]
        for qid, pid, query, text in group:
            q_terms = bm25.tokenizer.tokenize_stem(query)
            t_terms = bm25.tokenizer.tokenize_stem(text)
            q_tf = Counter(q_terms)
            t_tf = Counter(t_terms)
            score = bm25.score_inner(q_tf, t_tf)
            tfs_list.append((pid, t_tf))
            scores.append((pid, score))

        qid_scores.append((qid, scores))
        qid_tfs = qid, tfs_list
        save_qid_tfs(qid, qid_tfs)
    save_shallow_scores(qid_scores, shallow_score_save_path)


def main():
    job_no = sys.argv[1]
    split = "dev"
    c_log.info("Job %s", job_no)
    src_path = get_mmp_grouped_sorted_path(split, job_no)
    itr = tsv_iter(src_path)
    shallow_score_save_path = get_shallow_score_save_path(split, job_no)
    compute_base_score(itr, shallow_score_save_path)


if __name__ == "__main__":
    main()

