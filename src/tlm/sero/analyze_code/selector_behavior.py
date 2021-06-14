import sys
from collections import Counter

from cache import load_pickle_from
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def main():

    prediction_file = sys.argv[1]
    data = EstimatorPredictionViewer(prediction_file, [
                    "logits",
                    'is_valid_window',
                    'has_any_content',
                    'logits3_d',
                    'max_seg',
    ])

    raw_data = load_pickle_from(prediction_file)

    # Q: How many has too short content?
    counter = Counter()
    cnt = 0

    for batch in raw_data:
        maybe_batch_size = len(batch['has_any_content'])
        print('maybe_batch_size', maybe_batch_size)
        for batch_i in range(maybe_batch_size):
            def get_vector(name):
                return batch[name][batch_i]
            is_valid_window = get_vector("is_valid_window")
            max_seg = get_vector("max_seg")
            logits3_d = get_vector("logits3_d")

            # print(is_valid_window)
            print(max_seg)
            # print(has_any_content)
            for i in range(4):
                if is_valid_window[i]:
                    counter["appear_{}".format(i)] += 1
                    if max_seg == i:
                        counter["max_{}".format(i)] += 1

    for i in range(4):
        n_max = counter["max_{}".format(i)]
        n_appear = counter["appear_{}".format(i)]
        max_rate = n_max / n_appear
        print("{}\t{}".format(n_appear, max_rate))


if __name__ == "__main__":
    main()