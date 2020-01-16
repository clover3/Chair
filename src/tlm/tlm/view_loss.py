import os

from cpath import output_path
from misc_lib import average
from tlm.estimator_prediction_viewer import EstimatorPredictionViewerGosford


def view():
    filename = os.path.join(output_path, "nli_dev_loss.pickle")
    data = EstimatorPredictionViewerGosford(filename)

    loss_arr = []
    for inst_i, entry in enumerate(data):
        t = entry.get_vector("loss")
        loss_arr.append(float(t))

    print(len(loss_arr))
    print("avg:", average(loss_arr))

view()