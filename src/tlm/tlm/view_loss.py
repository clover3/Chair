import os

from misc_lib import average
from path import output_path
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def view():
    filename = os.path.join(output_path, "nli_dev_loss.pickle")
    data = EstimatorPredictionViewer(filename)

    loss_arr = []
    for inst_i, entry in enumerate(data):
        t = entry.get_vector("loss")
        loss_arr.append(float(t))

    print(len(loss_arr))
    print("avg:", average(loss_arr))

view()