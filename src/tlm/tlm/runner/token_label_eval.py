import sys

from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def eval(file_name):
    data = EstimatorPredictionViewer(file_name)
    for entry in data:
        entry.get_vector("masked_label_ids_label")
        entry.get_vector("is_test_inst")
        print(entry.get_vector("is_test_inst"), entry.get_vector("masked_lm_example_loss_label"))


if __name__ == "__main__":
    eval(sys.argv[1])