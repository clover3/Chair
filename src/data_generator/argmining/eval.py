import sys

import numpy as np

from task.metrics import eval_3label
from tf_util.enum_features import load_record
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def load_tfrecord(record_path):
    for feature in load_record(record_path):
        input_ids = feature["input_ids"].int64_list.value
        label_ids = feature["label_ids"].int64_list.value[0]
        yield input_ids, label_ids


def load_preditions(path):
    data = EstimatorPredictionViewer(path)
    for entry in data:
        input_ids = entry.get_vector("input_ids")
        logits = entry.get_vector("logits")
        yield input_ids, logits


def get_f1_score(tfrecord_path, prediction_path):
    tfrecord = list(load_tfrecord(tfrecord_path))
    predictions = list(load_preditions(prediction_path))
    golds, preds = zip(*compare(tfrecord, predictions))
    golds = golds[:len(preds)]
    all_result = eval_3label(preds, golds)
    f1 = sum([result['f1'] for result in all_result]) / 3
    return {"f1": f1}


def compare(tfrecord, predictions):
    for record, pred in zip(tfrecord, predictions):
        input_ids_r, label_ids = record
        input_ids_p, logits = pred
        if not np.all(input_ids_r == input_ids_p):
            print(input_ids_r)
            print(input_ids_p)
        assert np.all(input_ids_r == input_ids_p)
        pred_label = np.argmax(logits)
        yield label_ids, pred_label


if __name__ == "__main__":
    print(get_f1_score(sys.argv[1], sys.argv[2]))