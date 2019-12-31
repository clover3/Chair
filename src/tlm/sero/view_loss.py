import os
import pickle

from path import output_path
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from visualize.html_visual import HtmlVisualizer


def loss_view():
    filename = "sero_pred.pickle"
    p = os.path.join(output_path, filename)
    data = pickle.load(open(p, "rb"))
    print(data[0]["masked_lm_example_loss"].shape)
    print(data[0]["masked_input_ids"].shape)

    html_writer = HtmlVisualizer("sero_pred.html", dark_mode=False)

    data = EstimatorPredictionViewer(filename)
    for inst_i, entry in enumerate(data):
        if inst_i > 10:
            break
        losses = entry.get_vector("masked_lm_example_loss")
        print(losses)
        tokens = entry.get_tokens("masked_input_ids")
        cells = data.cells_from_tokens(tokens)
        row = []
        for idx, cell in enumerate(cells):
            row.append(cell)
            if len(row) == 20:
                html_writer.write_table([row])
                row = []

        html_writer.multirow_print(data.cells_from_anything(losses), 20)

if __name__ == '__main__':
    loss_view()