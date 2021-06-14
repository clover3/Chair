import sys
from collections import Counter

import numpy as np
from scipy.special import softmax

from cache import load_pickle_from
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def main():

    prediction_file = sys.argv[1]
    data = EstimatorPredictionViewer(prediction_file, [
                    "label_ids",
                    'is_valid_window',
                    'logits',
    ])

    raw_data = load_pickle_from(prediction_file)

    # Q: How many has too short content?
    counter = Counter()
    cnt = 0

    for batch in raw_data:
        try:
            label_ids = batch['label_ids']
            maybe_batch_size = len(batch['label_ids'])
            logit3d = np.reshape(batch['logits'], [maybe_batch_size, 4, -1])
            probs = softmax(logit3d)[:, :, 1]

            for batch_i in range(maybe_batch_size):
                label = label_ids[batch_i]
                if not label:
                    continue
                def get_vector(name):
                    return batch[name][batch_i]
                is_valid_window = get_vector("is_valid_window")

                max_seg = 0
                for i in range(4):
                    if is_valid_window[i]:
                        if probs[batch_i][i] > probs[batch_i][max_seg]:
                            max_seg = i

                # print(is_valid_window)
                # print(has_any_content)
                for i in range(4):
                    if is_valid_window[i]:
                        counter["appear_{}".format(i)] += 1
                        if probs[batch_i][i] > 0.1:
                            counter["pos_{}".format(i)] += 1

                seg_len = 0
                for i in range(4):
                    if is_valid_window[i]:
                        seg_len = i+1

                counter["{}/{}".format(max_seg, seg_len)] += 1
        except ValueError as e:
            print(e)

    for i in range(4):
        n_max = counter["max_{}".format(i)]
        n_appear = counter["pos_{}".format(i)]
        max_rate = n_max / n_appear if n_appear else 0
        print("{}\t{}".format(n_appear, max_rate))

    for seg_len in range(5):
        print("seg_len={}".format(seg_len))
        nums = []
        for j in range(4):
            n = counter["{}/{}".format(j, seg_len)]
            nums.append(n)
        print(nums)



if __name__ == "__main__":
    main()