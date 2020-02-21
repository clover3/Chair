import os
import sys
from functools import partial

import numpy as np
from scipy.stats import stats

from data_generator.argmining.eval import load_tfrecord, load_preditions, compare
from misc_lib import lmap, flatten


def prediction_path(run_name):
    path_prefix = "./output/ukp/" + run_name
    topic = "abortion"
    return path_prefix + "_" + topic

def get_existing_predictions(prefix, topic):
    path_prefix = "./output/ukp/"
    for i in range(10):
        run_name = prefix + "_{}".format(i)
        prediction_path = path_prefix + run_name + "_" + topic
        if os.path.exists(prediction_path):
            predictions = load_preditions(prediction_path)
            yield predictions



def main(prefix1, prefix2):
    topic = "abortion"
    tfrecord_path = "./data/ukp_tfrecord/dev_" + topic
    tfrecord = list(load_tfrecord(tfrecord_path))

    get_correctness_arr_fn = partial(get_correctness_arr, tfrecord)

    prediction_list_1 = list(get_existing_predictions(prefix1, topic))
    prediction_list_2 = list(get_existing_predictions(prefix2, topic))

    num_runs = min(len(prediction_list_1), len(prediction_list_2))
    prediction_list_1 = prediction_list_1[:num_runs]
    prediction_list_2 = prediction_list_2[:num_runs]

    c1 = flatten(lmap(get_correctness_arr_fn, prediction_list_1))
    c2 = flatten(lmap(get_correctness_arr_fn, prediction_list_2))

    print(len(c1))
    print(len(c2))

    _, p_value = stats.ttest_rel(c1, c2)
    print(p_value)


def get_correctness_arr(tfrecord, predictions):
    golds, preds = zip(*compare(tfrecord, predictions))
    correct = np.equal(golds, preds)
    return correct.astype(float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])