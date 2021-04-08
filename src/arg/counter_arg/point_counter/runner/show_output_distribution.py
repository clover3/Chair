import sys

from misc_lib import BinHistogram
from scipy_aux import logit_to_score_softmax
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def main():
    prediction_file = sys.argv[1]
    pred_data = EstimatorPredictionViewer(prediction_file)

    def bin_fn(score):
        return str(int(score * 100))

    bin = BinHistogram(bin_fn)
    for idx, e in enumerate(pred_data):
        score = logit_to_score_softmax(e.get_vector('logits'))
        bin.add(score)

    print(bin.counter.keys())
    for i in range(101):
        key = str(i)
        if key in bin.counter:
            print(key, bin.counter[key])

if __name__ == "__main__":
    main()