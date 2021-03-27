import sys

import numpy as np
import scipy.special

from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def load_preditions(path):
    data = EstimatorPredictionViewer(path)
    for entry in data:
        logits = entry.get_vector("logits")
        yield logits


def show(prediction_path):
    logits = list(load_preditions(prediction_path))
    logits_np = np.stack(logits)
    probs = scipy.special.softmax(logits_np, axis=1)
    preds = np.argmax(probs, axis=1)

    for label in [0,1,2]:
        n = np.count_nonzero(preds == label)
        print(label, n)


def main():
    return show(sys.argv[1])


if __name__ == "__main__":
    main()



