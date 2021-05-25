import json
import os
from typing import List, Iterable, Callable, Dict, Tuple, Set

from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap
from misc_lib import group_by, find_max_idx, TimeProfiler
from scipy_aux import get_logits_to_score_fn


class BestSegCollector:
    def __init__(self, info_dir, prediction_dir,
                 score_type):
        self.prediction_dir = prediction_dir
        self.info_dir = info_dir
        self.logits_to_score = get_logits_to_score_fn(score_type)

    def get_best_seg_info(self, job_id) -> Dict[Tuple[str, str], int]:
        info = json.load(open(os.path.join(self.info_dir, str(job_id) + ".info"), "r"))
        prediction_file = os.path.join(self.prediction_dir, str(job_id) + ".score")
        pred_data: List[Dict] = join_prediction_with_info(prediction_file, info)

        def get_score(entry):
            return self.logits_to_score(entry['logits'])

        qdid_grouped = group_by(pred_data, lambda d: (d['qid'], d['doc_id']))
        qdid_to_max_seg_idx: Dict[Tuple[str, str], int] = {}
        for qdi, entries in qdid_grouped.items():
            query_id, doc_id = qdi
            max_seg_idx = entries[find_max_idx(entries, get_score)]['seg_idx']
            qdid_to_max_seg_idx[query_id, doc_id] = max_seg_idx
        return qdid_to_max_seg_idx
