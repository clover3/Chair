import sys
from typing import List, Iterable, Dict, Tuple
from scipy.stats import pearsonr

from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter, enum_grouped2
from dataset_specific.msmarco.passage.path_helper import get_mmp_grouped_sorted_path
from dataset_specific.msmarco.passage.runner.build_ranked_list import read_scores
from list_lib import assert_length_equal
from misc_lib import select_first_second
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.compare_two_ranked_list import IndexedRankedList, ScoringModel, TermEffectMeasure
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_deep_model_score_path, \
    load_tfs_and_computed_base_scores, get_qtfs_save_path, load_qtf_index


def pearson_r_wrap(scores1: List[float], scores2: List[float]) -> float:
    r, p = pearsonr(scores1, scores2)
    return r


QID_PID_SCORE = Tuple


def load_deep_scores(job_no):
    scores_path = get_deep_model_score_path(job_no)
    quad_tsv_path = get_mmp_grouped_sorted_path(job_no)

    qid_pid: List[Tuple[str, str]] = list(select_first_second(tsv_iter(quad_tsv_path)))
    c_log.info("Reading deep model's scores")
    scores = read_scores(scores_path)
    assert_length_equal(scores, qid_pid)
    print("{} scores".format(len(scores)))
    items = [(qid, pid, score) for (qid, pid), score in zip(qid_pid, scores)]
    grouped: Iterable[List[QID_PID_SCORE]] = list(enum_grouped2(items))
    print("{} groups".format(len(grouped)))
    return grouped


def load_deep_and_shallow_scores(job_no) -> Dict[str, IndexedRankedList]:
    c_log.info("Loading tfs and base scores")
    tfs_and_shallow = load_tfs_and_computed_base_scores(job_no)
    deep_score_grouped_itr = load_deep_scores(job_no)

    irl_d: Dict[str, IndexedRankedList] = {}
    for s_entries, d_entries in zip(tfs_and_shallow, deep_score_grouped_itr):
        qid, _, _ = d_entries[0]

        if not len(s_entries) == len(d_entries):
            raise Exception()

        e_list = []
        for (pid, tfs, score_s), (qid_, pid, score_d) in zip(s_entries, d_entries):
            e = IndexedRankedList.Entry(
                doc_id=pid,
                deep_model_score=score_d,
                shallow_model_score_base=score_s,
                tfs=tfs
            )
            e_list.append(e)

        irl = IndexedRankedList(qid, e_list)
        irl_d[qid] = irl
    return irl_d


def main():
    # Measure difference of two given ranked list
    job_no = int(sys.argv[1])
    bm25 = get_bm25_mmp_25_01_01()
    sm = ScoringModel(bm25.core.k1, bm25.core.b, bm25.core.avdl, bm25.term_idf_factor)
    irl_d: Dict[str, IndexedRankedList] = load_deep_and_shallow_scores(job_no)
    tem = TermEffectMeasure(
        sm.get_updated_score_bm25,
        pearson_r_wrap,
        irl_d,
        load_qtf_index(job_no)
    )
    c_log.info("Done loading resources")
    gain_list = tem.term_effect_measure("when", "sunday")
    print(gain_list)
    c_log.info("Done evaluating effects")


if __name__ == "__main__":
    main()