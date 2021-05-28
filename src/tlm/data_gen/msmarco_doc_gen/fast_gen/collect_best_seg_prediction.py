import json
import os
from collections import defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set

from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap
from misc_lib import group_by, find_max_idx, TimeProfiler, get_first
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

    def get_best_seg_info_2d(self, job_id) -> Dict[str, Dict[str, int]]:
        qdid_to_max_seg_idx = self.get_best_seg_info(job_id)
        qdis: Dict[str, List[Tuple[str, str]]] = group_by(qdid_to_max_seg_idx.keys(), get_first)
        output = defaultdict(dict)
        for qid, entries in qdis:
            for qid_, doc_id in entries:
                output[qid][doc_id] = qdid_to_max_seg_idx[qid, doc_id]
        return output

