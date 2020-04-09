import numpy as np
from scipy.special import softmax

from tlm.estimator_prediction_viewer import EstimatorPredictionViewerGosford
from visualize.html_visual import HtmlVisualizer


def draw():
    name="pc_para_D_grad"
    data = EstimatorPredictionViewerGosford(name)
    html_writer = HtmlVisualizer(name + ".html", dark_mode=False)

    for inst_i, entry in enumerate(data):
        tokens = entry.get_tokens("input_ids")
        grad = entry.get_vector("gradient")
        m = min(grad)

        cells = data.cells_from_tokens(tokens)

        for i, cell in enumerate(cells):
            cells[i].highlight_score = min(abs(grad[i]) * 1e11, 255)
            cells[i].target_color = "B" if grad[i] > 0 else "R"
        print(entry.get_vector("logits"))
        prob = softmax(entry.get_vector("logits"))

        pred = np.argmax(prob)

        label = entry.get_vector("labels")
        html_writer.write_paragraph("Label={} / Pred={}".format(str(label), pred))
        html_writer.multirow_print(cells)



draw()