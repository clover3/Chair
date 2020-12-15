
from typing import List, Dict, Tuple

from arg.perspectives.load import evidence_gold_dict_str_qid
from arg.qck.decl import QCKQuery, QCKCandidate
from list_lib import lmap
from misc_lib import get_f1, average


def get_precision_recall(input_entries: List[Tuple[QCKQuery, List[QCKCandidate]]]) -> Dict:
    gold_dict: Dict[str, List[int]] = evidence_gold_dict_str_qid()

    all_scores = []
    for query, ranked_list in input_entries:
        e_id_list = lmap(QCKCandidate.get_id, ranked_list)
        gold_id = gold_dict[query.query_id]

        tp = 0
        for e_id in e_id_list:
            if e_id in gold_id:
                tp += 1

        precision = tp / len(e_id_list) if len(e_id_list) else 1
        recall = tp / len(gold_id) if len(gold_id) else 1
        f1 = get_f1(precision, recall)
        per_score = {
            'precision': precision,
            'recall': recall,
        }
        all_scores.append(per_score)

    average_scores = {}
    for metric in ['precision', 'recall']:
        average_scores[metric] = average([e[metric] for e in all_scores])

    average_scores['f1'] = get_f1(average_scores['precision'], average_scores['recall'])
    return average_scores
