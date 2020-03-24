import os
import random

import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from cache import load_from_pickle
from cpath import output_path
from crs.contradiction_pair.get_val_data import get_label_as_or
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def show():
    p = os.path.join(output_path, "pair_eval_1")
    d = EstimatorPredictionViewer(p)

    pred0 = load_from_pickle("cont_model_0")

    labels = get_label_as_or()
    entries = []
    entries2 = []
    for idx, entry in enumerate(d):
        logits = entry.get_vector("logits")
        probs = softmax(logits)
        pred = np.argmax(probs)
        entries.append(probs[2])
        entries2.append(random.random())
        #print(idx, pred==labels[idx], pred, labels[idx], probs)

    labels_binary = list([t==2 for t in labels])
    print(roc_auc_score(labels, entries))
    print(roc_auc_score(labels, entries2))
    print(roc_auc_score(labels, pred0))



if __name__ == "__main__":
    show()

