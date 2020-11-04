import sys

from scipy.special import softmax

from tab_print import print_table
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def main():
    data = EstimatorPredictionViewer(sys.argv[1])
    rows = []
    for e in data:
        label_ids = e.get_vector("label_ids")
        logits = e.get_vector("logits")
        probs = softmax(logits, -1)

        predict_label = 1 if probs[1] > 0.5 else 0

        decision = "Y" if predict_label == label_ids else "N"

        row = [label_ids, probs[1], decision]
        rows.append(row)

    rows.sort(key=lambda x: x[1], reverse=True)
    print_table(rows)


if __name__ == "__main__":
    main()