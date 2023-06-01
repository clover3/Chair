import dataclasses
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple

from krovetzstemmer import Stemmer

from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter, enum_grouped2
from dataset_specific.msmarco.passage.path_helper import get_mmp_grouped_sorted_path
from dataset_specific.msmarco.passage.runner.build_ranked_list import read_scores
from list_lib import assert_length_equal
from misc_lib import select_first_second
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import TermEffectPerQuery
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_deep_model_score_path, \
    read_shallow_scores, load_mmp_tfs, read_shallow_score_per_qid, read_deep_score_per_qid
import psutil


def print_cur_memory():
    process = psutil.Process()
    byte_used = process.memory_info().rss
    gb_used = byte_used / 1024 / 1024 / 1024
    c_log.info("{0:.2f} gb used".format(gb_used))


class IndexedRankedList:
    @dataclass
    class Entry:
        doc_id: str
        deep_model_score: float
        shallow_model_score_base: float
        tfs: Counter

        def get_dl(self):
            dl = sum(self.tfs.values())
            return dl

    def __init__(self, qid, entries):
        self.entries: List[IndexedRankedList.Entry] = entries
        self.qid = qid

        inv_index = defaultdict(list)
        for idx, e in enumerate(self.entries):
            for t in e.tfs:
                inv_index[t].append(idx)

        self.inv_index = inv_index

    def get_entries_w_term(self, term) -> List[int]:
        return self.inv_index[term]

    def get_shallow_model_base_scores(self):
        return [e.shallow_model_score_base for e in self.entries]

    def get_deep_model_scores(self):
        return [e.deep_model_score for e in self.entries]


class TermEffectMeasure:
    def __init__(
            self,
            get_updated_score_fn: Callable[[str, str, IndexedRankedList.Entry], float],
            get_irl_fn: Callable[[str], IndexedRankedList],
            q_inv_index: Dict[str, List[str]]
    ):
        self.get_updated_score = get_updated_score_fn
        self.get_irl_fn: Callable[[str], IndexedRankedList] = get_irl_fn
        self.q_inv_index = q_inv_index
        self.stemmer = Stemmer()

    def _get_qid_for_q_term(self, q_term) -> List[str]:
        return self.q_inv_index[q_term]

    def _get_ranked_list(self, qid):
        return self.get_irl_fn(qid)

    def term_effect_measure(self, q_term, d_term) -> List[TermEffectPerQuery]:
        q_term = self.stemmer.stem(q_term)
        d_term = self.stemmer.stem(d_term)
        query_w_q_term: List[str] = self._get_qid_for_q_term(q_term)
        output = []
        for qid in query_w_q_term:
            ranked_list: IndexedRankedList = self._get_ranked_list(qid)
            old_scores: List[float] = ranked_list.get_shallow_model_base_scores()

            entry_indices = ranked_list.get_entries_w_term(d_term)
            # print("{} entries affected ".format(len(entry_indices)))
            changes = []
            for entry_idx in entry_indices:
                entry = ranked_list.entries[entry_idx]
                new_score: float = self.get_updated_score(q_term, d_term, entry)
                changes.append((entry_idx, new_score))

            target_scores = ranked_list.get_deep_model_scores()
            per_query = TermEffectPerQuery(target_scores, old_scores, changes)
            output.append(per_query)
        return output


class ScoringModel:
    def __init__(self, k1, b, avdl, get_qtw):
        self.k1 = k1
        self.b = b
        self.avdl = avdl
        self.get_qtw = get_qtw

    def get_updated_score_bm25(
            self, q_term: str, d_term: str, entry: IndexedRankedList.Entry) -> float:
        k1 = self.k1
        b = self.b
        avdl = self.avdl

        old_tf = entry.tfs[q_term]
        new_tf = entry.tfs[q_term] + entry.tfs[d_term]
        dl = entry.get_dl()
        qtw = self.get_qtw(q_term)

        def tf_factor(tf):
            denom = tf + k1 * tf
            nom = tf + k1 * ((1 - b) + b * dl / avdl)
            return denom / nom

        delta = (tf_factor(new_tf) - tf_factor(old_tf)) * qtw
        return entry.shallow_model_score_base + delta


QID_PID_SCORE = Tuple


def load_deep_scores(job_no) -> List[List[QID_PID_SCORE]]:
    scores_path = get_deep_model_score_path(job_no)
    quad_tsv_path = get_mmp_grouped_sorted_path(job_no)
    qid_pid: List[Tuple[str, str]] = list(select_first_second(tsv_iter(quad_tsv_path)))
    scores = read_scores(scores_path)
    assert_length_equal(scores, qid_pid)
    items = [(qid, pid, score) for (qid, pid), score in zip(qid_pid, scores)]
    grouped: List[List[QID_PID_SCORE]] = list(enum_grouped2(items))
    return grouped


class IRLProxy:
    @dataclasses.dataclass
    class Entry:
        doc_id: str
        deep_model_score: float
        shallow_model_score_base: float

    def __init__(self, job_no):
        pass
        # c_log.info("Loading deep scores")
        # deep_score_grouped: List[List[QID_PID_SCORE]] = load_deep_scores(job_no)
        # c_log.info("Loading shallow scores")
        # shallow_scores: List[Tuple[str, List[Tuple[str, float]]]] = read_shallow_scores(job_no)
        # assert_length_equal(deep_score_grouped, shallow_scores)
        #
        # qid_to_score_list = {}
        # for deep_score_group, shallow_score_group in zip(deep_score_grouped, shallow_scores):
        #     qid_s, entries = shallow_score_group
        #     assert_length_equal(deep_score_group, entries)
        #     e_list: List[IRLProxy.Entry] = []
        #     for e_d, e_s in zip(deep_score_group, entries):
        #         qid_d, pid_d, score_d = e_d
        #         pid_s, score_s = e_s
        #         assert qid_s == qid_d
        #         assert pid_d == pid_s
        #         e = IRLProxy.Entry(
        #             doc_id=pid_s,
        #             deep_model_score=score_d,
        #             shallow_model_score_base=score_s,
        #         )
        #         e_list.append(e)
        #     qid_to_score_list[qid_s] = e_list
        # self.qid_to_score_list = qid_to_score_list

    def get_irl(self, qid) -> IndexedRankedList:
        qid_, pid_tfs = load_mmp_tfs(qid)
        qid_s, pid_scores_s = read_shallow_score_per_qid(qid)
        qid_d, pid_scores_d = read_deep_score_per_qid(qid)

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

        return IndexedRankedList(qid, e_out_list)