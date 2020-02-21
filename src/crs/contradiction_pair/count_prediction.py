import os
import pickle
import sys

from cpath import output_path
from crs.contradiction_pair.pair_prediction_analysis import count_contradiction
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def count_and_save(j):
    p = os.path.join(output_path, "clueweb12_13B_pair", "nli_prediction_{}".format(j))
    d = EstimatorPredictionViewer(p)
    r = count_contradiction(d)
    p = os.path.join(output_path, "clueweb12_13B_pair_summary_{}".format(j))
    pickle.dump(r, open(p, "wb"))


if __name__ == "__main__":
    count_and_save(int(sys.argv[1]))
