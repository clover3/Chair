import sys

from arg.counter_arg.point_counter.sklearn_metrics import get_ap
from cache import load_pickle_from
from list_lib import lmap
from scipy_aux import logit_to_score_softmax
from tab_print import print_table
from tlm.estimator_prediction_viewer import flatten_batches


def get_ap_from_file_path(input_path):
    tf_prediction_data = load_pickle_from(input_path)
    tf_prediction_data = flatten_batches(tf_prediction_data)
    logits = tf_prediction_data["logits"]
    label_ids = tf_prediction_data["label_ids"]

    scores = lmap(logit_to_score_softmax, logits)

    assert len(scores) == len(label_ids)
    return get_ap(label_ids, scores)


def main():
    input_path_format = sys.argv[1]
    st = int(sys.argv[2])
    step = int(sys.argv[3])
    ed = int(sys.argv[4])

    rows = []
    for i in range(st, ed, step):
        s = get_ap_from_file_path(input_path_format.format(i))
        row = [i, s]
        rows.append(row)
    print_table(rows)


if __name__ == "__main__":
    main()