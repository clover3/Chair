import os
import pickle
import sys

from tlm.estimator_prediction_viewer import flatten_batches


def load_vectors(file_path):
    p = os.path.join(file_path)
    data = pickle.load(open(p, "rb"))
    keys = list(data[0].keys())
    vectors = flatten_batches(data)
    any_key = keys[0]
    data_len = len(vectors[any_key])
    return vectors


def do(file_path, out_path):
    vectors = load_vectors(file_path)
    threshold = 2000

    out_items = []
    for i in range(len(vectors)):
        score = vectors["overlap_score"][i]
        if score > threshold:
            e = []
            for key in ["input_ids", "masked_input_ids", "masked_lm_example_loss"]:
                e.append(vectors[key][i])
            out_items.append(e)

    pickle.dump(out_items, open(out_path, "wb"))


if __name__ == '__main__':
    do(sys.argv[1], sys.argv[2])