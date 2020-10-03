from typing import List, Dict, Tuple, NamedTuple

import scipy.special

from arg.qck.decl import QCKQuery, KDP, QCKCandidate
from arg.qck.prediction_reader import qck_convert_map, load_combine_info_jsons
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap
from misc_lib import group_by, get_first


def load_and_group_predictions(info_path, pred_path):
    info = load_combine_info_jsons(info_path, qck_convert_map, False)
    predictions: List[Dict] = join_prediction_with_info(pred_path, info)
    out_entries: List[QCKOutEntry] = lmap(QCKOutEntry.from_dict, predictions)


def doc_value(score, score_baseline, gold):
    error_baseline = gold - score_baseline  # -1 ~ 1
    error = gold - score
    improvement = abs(error_baseline) - abs(error)
    return improvement


class QCKOutEntry(NamedTuple):
    logits: List[float]
    query: QCKQuery
    candidate: QCKCandidate
    kdp: KDP

    @classmethod
    def from_dict(cls, d):
        return QCKOutEntry(d['logits'], d['query'], d['candidate'], d['kdp'])


class DocValueParts(NamedTuple):
    logits: List[float]
    query: QCKQuery
    candidate: QCKCandidate
    kdp: KDP
    score: float
    value: float


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


