import os
from collections import Counter

import numpy as np
from scipy.special import softmax

from cpath import output_path
from crs.contradiction_pair.datagen_common import save_to_tfrecord
from crs.contradiction_pair.pair_prediction_analysis import load_data


def select_data(f_do_select, f_even_sample, data):
    max_select = 10000

    num_insts = Counter()
    for entry in data:
        logits = entry.get_vector("logits")
        input_ids = entry.get_vector("input_ids")

        probs = softmax(logits)
        pred = np.argmax(probs)
        if probs[pred] > 0.5 or not f_do_select:

            if not f_even_sample or num_insts[pred] < max_select:
                nd = {
                    'input_ids':input_ids,
                    'label': pred,
                }
                num_insts[pred] += 1
                yield nd

    print(num_insts)


def runner():
    f_do_select = True
    f_even_sample = True

    data_name = "weather"
    data, pickle_path = load_data(data_name)
    out_path = os.path.join(output_path, "self_training_weather_even")

    r = select_data(f_do_select, f_even_sample, data)

    save_to_tfrecord(r, out_path)


if __name__ == "__main__":
    runner()




