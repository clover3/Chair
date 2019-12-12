import numpy as np
import scipy.special

from misc_lib import lmap, average
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from visualize.html_visual import HtmlVisualizer


def probabilty(scores, amp):
    alpha = 0
    seq_length = len(scores)
    prob = scipy.special.softmax(scores * amp)

    p1 = np.ones_like(prob) / seq_length * alpha
    p2 = prob * (1-alpha)
    final_p = p1 + p2
    return final_p

def geo_mean_prob(scores, amp):
    alpha = 0
    scores = scores + 1
    row_sum = np.sum(scores)
    row_sum = np.expand_dims(row_sum, 0)
    prob = np.divide(scores, row_sum)
    #prob = scipy.special.softmax(scores * amp)
    final_p = prob * (1-alpha)
    return final_p


def doit():
    filename = "tlm_view.pickle"

    data = EstimatorPredictionViewer(filename)
    amp = 40
    html_writer = HtmlVisualizer("tlm_view{}.html".format(amp), dark_mode=False)

    for inst_i, entry in enumerate(data):
        if inst_i > 100:
            break
        tokens = entry.get_mask_resolved_input_mask_with_input()
        scores = entry.get_vector("priority_score")
        prob_scores = probabilty(scores, amp)
        prob_strs = ["{:06.6f}".format(v*1000) for v in prob_scores]

        def normalize(prob):
            # 0-> Good
            # -1 -> Bad
            return (-prob + 0.1) * 100

        norm_scores = lmap(normalize, prob_scores)
        cells = data.cells_from_tokens(tokens, norm_scores)
        cells2 = data.cells_from_anything(scores, norm_scores)
        cells3 = data.cells_from_anything(prob_strs, norm_scores)

        row1 = []
        row2 = []
        row3 = []
        for idx, cell in enumerate(cells):
            row1.append(cell)
            row2.append(cells2[idx])
            row3.append(cells3[idx])
            if len(row1) == 20:
                html_writer.write_table([row1, row2, row3])
                row1 = []
                row2 = []
                row3 = []
        html_writer.write_headline("")


def per_doc_score():
    filename = "tlm_view.pickle"
    html_writer = HtmlVisualizer("per_doc_score.html", dark_mode=False)

    data = EstimatorPredictionViewer(filename)
    amp = 20
    small_threshold = 40
    for inst_i, entry in enumerate(data):
        if inst_i > 1000:
            break
        scores = entry.get_vector("priority_score")

        tokens = entry.get_mask_resolved_input_mask_with_input()
        cells = data.cells_from_tokens(tokens)
        if len(cells) < small_threshold:
            continue
        avg_score = average(scores)
        if -0.11 > avg_score > -0.30:
            continue
        print(average(scores))
        html_writer.write_headline(avg_score)
        rows = []
        row = []
        for idx, cell in enumerate(cells):
            row.append(cell)
            if len(row) == 20:
                html_writer.write_table([row])
                row = []



if __name__ == '__main__':
    doit()