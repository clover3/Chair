import sys

from arg.counter_arg.point_counter.sklearn_metrics import get_auc, get_ap
from cache import load_pickle_from
from list_lib import lmap
from scipy_aux import logit_to_score_softmax
from tab_print import print_table
from tlm.estimator_prediction_viewer import flatten_batches


def main():
    input_path = sys.argv[1]
    tf_prediction_data = load_pickle_from(input_path)
    tf_prediction_data = flatten_batches(tf_prediction_data)
    logits = tf_prediction_data["logits"]
    label_ids = tf_prediction_data["label_ids"]

    scores = lmap(logit_to_score_softmax, logits)

    assert len(scores) == len(label_ids)
    print("{} data points".format(len(scores)))
    todo = [(get_auc, "auc"), (get_ap, "ap")]
    rows = []
    for metric_fn, metric_name in todo:
        score = metric_fn(label_ids, scores)
        row = [metric_name, score]
        rows.append(row)

    print_table(rows)


if __name__ == "__main__":
    main()
