from typing import List, Dict, Tuple, NamedTuple

import scipy.special

from arg.qck.decl import QCKQuery, KDP, QCKCandidate, qck_convert_map, QCKOutEntry
from arg.qck.prediction_reader import load_combine_info_jsons
from list_lib import lmap
from misc_lib import group_by, get_first
from tlm.estimator_output_reader import join_prediction_with_info


def load_and_group_predictions(info_path, pred_path) -> List[QCKOutEntry]:
    info = load_combine_info_jsons(info_path, qck_convert_map, False)
    predictions: List[Dict] = join_prediction_with_info(pred_path, info)
    out_entries: List[QCKOutEntry] = lmap(QCKOutEntry.from_dict, predictions)
    return out_entries


def doc_value(score, score_baseline, gold):
    error_baseline = gold - score_baseline  # -1 ~ 1
    error = gold - score
    improvement = abs(error_baseline) - abs(error)
    return improvement


class DocValueParts(NamedTuple):
    logits: List[float]
    query: QCKQuery
    candidate: QCKCandidate
    kdp: KDP
    score: float
    value: float


class DocValueParts2(NamedTuple):
    logits: List[float]
    query: QCKQuery
    candidate: QCKCandidate
    kdp: KDP
    score: float
    label: int
    init_score: float

    @classmethod
    def init(self, e: QCKOutEntry, label, base_score):
        score: float = logit_to_score_softmax(e.logits)
        init_pred: bool = base_score > 0.5
        new_pred: bool = score > 0.5
        return DocValueParts2(logits=e.logits,
                              query=e.query,
                              candidate=e.candidate,
                              kdp=e.kdp,
                              score=score,
                              label=label,
                              init_score=base_score,
                              )


def logit_to_score_softmax(logit):
    return scipy.special.softmax(logit)[1]


def get_doc_value_parts(out_entries: List[QCKOutEntry],
                        baseline_score_d: Dict[Tuple[str, str], float],
                        gold: Dict[str, List[str]]) -> List[DocValueParts]:

    def get_qid(entry: QCKOutEntry):
        return entry.query.query_id

    def get_candidate_id(entry: QCKOutEntry):
        return entry.candidate.id

    print("baseline has {} entries".format(len(baseline_score_d.keys())))
    print("baseline has {} qids".format(len(group_by(baseline_score_d.keys(), get_first))))
    not_found_baseline = set()
    dvp_entries = []
    for qid, entries in group_by(out_entries, get_qid).items():
        gold_candidate_ids: List[str] = gold[qid]
        candi_grouped = group_by(entries, get_candidate_id)
        print(qid, len(candi_grouped))

        for candidate_id, sub_entries in candi_grouped.items():
            try:
                key = qid, candidate_id
                base_score: float = baseline_score_d[key]
                label = candidate_id in gold_candidate_ids
                sub_entries: List[QCKOutEntry] = sub_entries
                for e in sub_entries:
                    score: float = logit_to_score_softmax(e.logits)
                    value: float = doc_value(score, base_score, int(label))
                    dvp = DocValueParts(e.logits, e.query, e.candidate, e.kdp,
                                        score, value)
                    dvp_entries.append(dvp)
            except KeyError:
                not_found_baseline.add(key)
                if len(not_found_baseline) > 10:
                    raise KeyError()
    return dvp_entries


def doc_value_type(score: float, base_score: float, label: int):
    init_pred: bool = base_score > 0.5
    new_pred: bool = score > 0.5

    def get_decision_change():
        if init_pred != new_pred:
            if new_pred == bool(label):
                return "decision_change_good"
            else:
                return "decision_change_bad"
        else:
            return "no_change"

    change_direction = "up" if score > base_score else "down"
    return label, init_pred, score-base_score, get_decision_change()




def get_doc_value_parts2(out_entries: List[QCKOutEntry],
                         baseline_score_d: Dict[Tuple[str, str], float],
                         gold: Dict[str, List[str]]) -> List[DocValueParts2]:

    def get_qid(entry: QCKOutEntry):
        return entry.query.query_id

    def get_candidate_id(entry: QCKOutEntry):
        return entry.candidate.id

    print("baseline has {} entries".format(len(baseline_score_d.keys())))
    print("baseline has {} qids".format(len(group_by(baseline_score_d.keys(), get_first))))
    not_found_baseline = set()
    dvp_entries = []
    for qid, entries in group_by(out_entries, get_qid).items():
        gold_candidate_ids: List[str] = gold[qid]
        candi_grouped = group_by(entries, get_candidate_id)
        print(qid, len(candi_grouped))

        for candidate_id, sub_entries in candi_grouped.items():
            try:
                key = qid, candidate_id
                base_score: float = baseline_score_d[key]
                label = candidate_id in gold_candidate_ids
                sub_entries: List[QCKOutEntry] = sub_entries
                for e in sub_entries:
                    dvp = DocValueParts2.init(e, label, base_score)
                    dvp_entries.append(dvp)
            except KeyError:
                not_found_baseline.add(key)
                if len(not_found_baseline) > 32:
                    print(not_found_baseline)
                    raise KeyError()
    return dvp_entries


