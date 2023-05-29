from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Callable, Dict


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
            fidelity_fn: Callable[[List[float], List[float]], float],
            irl_d: Dict[str, IndexedRankedList],
            q_inv_index: Dict[str, List[str]]
    ):
        self.get_updated_score = get_updated_score_fn
        self.fidelity_fn: Callable[[List[float], List[float]], float] = fidelity_fn
        self.irl_d: Dict[str, IndexedRankedList] = irl_d
        self.q_inv_index = q_inv_index

    def _get_qid_for_q_term(self, q_term) -> List[str]:
        return self.q_inv_index[q_term]

    def _get_ranked_list(self, qid):
        return self.irl_d[qid]

    def term_effect_measure(self, q_term, d_term):
        query_w_q_term: List[str] = self._get_qid_for_q_term(q_term)
        gain_list: List[float] = []
        for qid in query_w_q_term:
            ranked_list: IndexedRankedList = self._get_ranked_list(qid)
            old_scores: List[float] = ranked_list.get_shallow_model_base_scores()

            entry_indices = ranked_list.get_entries_w_term(d_term)
            print("{} entries affected ".format(len(entry_indices)))
            new_scores: List[float] = list(old_scores)
            for entry_idx in entry_indices:
                entry = ranked_list.entries[entry_idx]
                new_score: float = self.get_updated_score(q_term, d_term, entry)
                new_scores[entry_idx] = new_score

            target_scores = ranked_list.get_deep_model_scores()
            fidelity_before = self.fidelity_fn(target_scores, old_scores)
            fidelity_after = self.fidelity_fn(target_scores, new_scores)
            gain = fidelity_after - fidelity_before
            gain_list.append(gain)
        return gain_list


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


