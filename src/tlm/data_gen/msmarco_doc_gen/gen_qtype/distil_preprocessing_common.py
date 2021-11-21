import os
import pickle
from typing import Dict, Tuple, List

from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.job_runner import WorkerInterface
from tf_util.enum_features import load_record
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def parse_estimator_prediction(prediction_path):
    itr = EstimatorPredictionViewer(prediction_path)
    out_d = {}
    for e in itr:
        data_id = int(e.get_vector("data_id")[0])
        logits = e.get_vector("logits")
        assert len(logits) == 1
        out_d[data_id] = logits[0]
    return out_d


def parse_tfrecord(tfrecord_path)  -> Dict[int, Tuple[List[int], List[int]]]:
    def values(feature, key):
        return feature[key].int64_list.value

    out_d = {}
    for record in load_record(tfrecord_path):
        input_ids = values(record, "input_ids")
        data_id = int(values(record, "data_id")[0])
        query, doc = split_p_h_with_input_ids(input_ids, input_ids)
        out_d[data_id] = query, doc
    return out_d


class ScoreTokenJoin(WorkerInterface):
    def __init__(self, tfrecord_dir, prediction_dir, out_dir):
        self.out_dir = out_dir
        self.prediction_dir = prediction_dir
        self.tfrecord_dir = tfrecord_dir

    def work(self, job_id):
        # Data ID -> score
        tfrecord_path = os.path.join(self.tfrecord_dir, str(job_id))
        if not os.path.exists(tfrecord_path):
            return
        # Data ID -> [Query Tokens, Doc Tokens]
        tokens_d = parse_tfrecord(tfrecord_path)
        prediction_path = os.path.join(self.prediction_dir, str(job_id))
        prediction_scores: Dict[int, float] = parse_estimator_prediction(prediction_path)
        save_entries = []
        for data_id in prediction_scores.keys():
            score = prediction_scores[data_id]
            q_tokens, d_tokens = tokens_d[data_id]
            d = {
                'q_tokens': q_tokens,
                'd_tokens': d_tokens,
                'score': score,
                'data_id': data_id
            }
            save_entries.append(d)

        save_path = os.path.join(self.out_dir, str(job_id))
        pickle.dump(save_entries, open(save_path, "wb"))