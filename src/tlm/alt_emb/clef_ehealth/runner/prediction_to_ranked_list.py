import sys

from cache import load_from_pickle
from tlm.alt_emb.clef_ehealth.prediction_to_ranked_list import prediction_to_ranked_list
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def main(pred_path, info_path, output_path):
    pred_data = EstimatorPredictionViewer(pred_path)
    info = load_from_pickle(info_path)
    prediction_to_ranked_list(pred_data, info, output_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
