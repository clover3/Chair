import sys

import numpy as np

from evals.tfrecord import load_tfrecord
from task.metrics import eval_3label, eval_2label
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def load_preditions(path):
    data = EstimatorPredictionViewer(path)
    for entry in data:
        input_ids = entry.get_vector("input_ids")
        logits = entry.get_vector("logits")
        yield input_ids, logits


def get_f1_score(tfrecord_path, prediction_path, n_label=3):
    tfrecord = list(load_tfrecord(tfrecord_path))
    predictions = list(load_preditions(prediction_path))
    golds, preds = zip(*compare(tfrecord, predictions))
    golds = golds[:len(preds)]
    if n_label == 3:
        all_result = eval_3label(preds, golds)
    elif n_label == 2:
        all_result = eval_2label(preds, golds)
    else:
        assert False

    f1 = sum([result['f1'] for result in all_result]) / n_label
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