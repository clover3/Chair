from typing import List, Tuple
from dataset_specific.msmarco.passage.passage_resource_loader import enum_grouped2
from table_lib import tsv_iter
from dataset_specific.msmarco.passage.path_helper import get_mmp_grouped_sorted_path
from adhoc.eval_helper.line_format_to_trec_ranked_list import read_scores
from list_lib import assert_length_equal
from misc_lib import select_first_second, path_join
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import read_shallow_score_per_qid, read_deep_score_per_qid, get_deep_model_score_path, get_tfs_save_path
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import IndexedRankedList, \
    QID_PID_SCORE, IRLProxyIF
from trec.trec_parse import scores_to_ranked_list_entries, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry
import sys




def load_deep_scores(score_dir, split, job_no) -> List[List[QID_PID_SCORE]]:
    scores_path = path_join(score_dir, f"{job_no}.scores")
    quad_tsv_path = get_mmp_grouped_sorted_path(split, job_no)
    qid_pid: List[Tuple[str, str]] = list(select_first_second(tsv_iter(quad_tsv_path)))
    scores = read_scores(scores_path)
    items = [(qid, pid, score) for (qid, pid), score in zip(qid_pid, scores)]
    grouped: List[List[QID_PID_SCORE]] = list(enum_grouped2(items))
    return grouped


def get_queries(split, job_no):
    quad_tsv_path = get_mmp_grouped_sorted_path(split, job_no)

    queries = []
    seen_qid = set()
    for qid, _pid, query, _doc in tsv_iter(quad_tsv_path):
        if qid not in seen_qid:
            queries.append((qid, query))
            seen_qid.add(qid)
    return queries


def main():
    job_no = int(sys.argv[1])
    split = "train"
    save_path = f"output/msmarco/passage/mmp_train_split_ranked_list/{job_no}.txt"
    query_save_path = f"output/msmarco/passage/mmp_train_split_queries/{job_no}.txt"
    score_dir = "output/msmarco/passage/mmp_train_split_all_scores_tf"
    grouped = load_deep_scores(score_dir, split, job_no)

    all_entries = []
    seen_qid = set()
    for group in grouped:
        qid = group[0][0]
        scores = [(pid, score) for qid, pid, score in group]
        entries = scores_to_ranked_list_entries(scores, "ce", qid)
        all_entries.extend(entries)
        seen_qid.add(qid)
    write_trec_ranked_list_entry(all_entries, save_path)

    queries = get_queries(split, job_no)
    save_tsv(queries, query_save_path)


if __name__ == "__main__":
    main()
