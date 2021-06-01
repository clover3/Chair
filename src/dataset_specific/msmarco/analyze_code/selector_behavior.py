import sys

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

    # Q: How many has too short content?
    cnt = 0
    for e in data:
        is_valid_window = e.get_vector("is_valid_window")
        max_seg = e.get_vector("max_seg")
        logits3_d = e.get_vector("logits3_d")

        # print(is_valid_window)
        print(max_seg, logits3_d, is_valid_window)
        # print(has_any_content)
        cnt += 1
        if cnt > 100:
            break


if __name__ == "__main__":
    main()