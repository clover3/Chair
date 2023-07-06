from typing import List, Tuple

from cache import load_pickle_from
from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter, enum_grouped2
from dataset_specific.msmarco.passage.path_helper import get_mmp_grouped_sorted_path
from dataset_specific.msmarco.passage.runner.build_ranked_list import read_scores
from list_lib import assert_length_equal
from misc_lib import select_first_second
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import load_mmp_tfs, \
    read_shallow_score_per_qid, read_deep_score_per_qid, get_deep_model_score_path, get_tfs_save_path
import psutil

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import IndexedRankedList, \
    QID_PID_SCORE, IRLProxyIF


def print_cur_memory():
    process = psutil.Process()
    byte_used = process.memory_info().rss
    gb_used = byte_used / 1024 / 1024 / 1024
    c_log.info("{0:.2f} gb used".format(gb_used))


def load_deep_scores(split, job_no) -> List[List[QID_PID_SCORE]]:
    scores_path = get_deep_model_score_path(split, job_no)
    quad_tsv_path = get_mmp_grouped_sorted_path(split, job_no)
    qid_pid: List[Tuple[str, str]] = list(select_first_second(tsv_iter(quad_tsv_path)))
    scores = read_scores(scores_path)
    assert_length_equal(scores, qid_pid)
    items = [(qid, pid, score) for (qid, pid), score in zip(qid_pid, scores)]
    grouped: List[List[QID_PID_SCORE]] = list(enum_grouped2(items))
    return grouped


# IRL: IndexedRankedList
# The cache exploits the fact (q_term, d_term) is sorted so that the same q_terms
# are repeated and it will access the same qids
class IRLProxy(IRLProxyIF):
    def __init__(self, q_term: str, profile):
        self.q_term = q_term
        self.cached = {}
        self.profile = profile
        self.read_shallow_score = read_shallow_score_per_qid
        self.read_deep_score = read_deep_score_per_qid
        self.get_tfs_save_path = get_tfs_save_path

    def set_new_q_term(self, q_term):
        if q_term == self.q_term:
            pass
        else:
            c_log.info("Reset cache ({}->{})".format(self.q_term, q_term))
            self.q_term = q_term
            self.cached = {}

    def get_irl(self, qid) -> IndexedRankedList:
        if qid in self.cached:
            return self.cached[qid]

        # c_log.debug("Loading for qid=%s", qid)
        qid_, pid_tfs = load_pickle_from(self.get_tfs_save_path(qid))
        qid_s, pid_scores_s = self.read_shallow_score(qid)
        qid_d, pid_scores_d = self.read_deep_score(qid)

        assert qid_ == qid_s
        assert qid_ == qid_d
        assert_length_equal(pid_tfs, pid_scores_s)
        assert_length_equal(pid_tfs, pid_scores_d)

        e_out_list = []
        for i in range(len(pid_tfs)):
            pid_, tfs = pid_tfs[i]
            pid_s, shallow_model_score_base = pid_scores_s[i]
            pid_d, deep_model_score = pid_scores_d[i]
            assert pid_ == pid_d
            e_out = IndexedRankedList.Entry(
                doc_id=pid_,
                deep_model_score=deep_model_score,
                shallow_model_score_base=shallow_model_score_base,
                tfs=tfs
            )
            e_out_list.append(e_out)

        irl = IndexedRankedList(qid, e_out_list)
        self.cached[qid] = irl
        return irl
