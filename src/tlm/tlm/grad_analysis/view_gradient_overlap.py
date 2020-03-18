import os
import pickle

import math
import numpy as np

from cpath import output_path
from list_lib import lmap
from misc_lib import IntBinAverage, average
from tlm.estimator_prediction_viewer import EstimatorPredictionViewerGosford
from tlm.token_utils import is_mask
from visualize.html_visual import HtmlVisualizer, Cell


def compare_grad_overlap():
    filename = "gradient_overlap_4K.pickle"
    data = EstimatorPredictionViewerGosford(filename)

    filename2 = "ukp_lm_overlap.pickle"
    data2 = EstimatorPredictionViewerGosford(filename2)

    scores1 = data.vectors["overlap_score"]
    scores2 = data2.vectors["overlap_score"]
    print(scores1.shape)

    def summary(scores):
        score_avg = average(scores)
        score_std = np.std(scores)
        print(score_avg, score_std)

    summary(scores1)
    summary(scores2)


def view_grad_overlap_per_mask():
    filename = "ukp_lm_probs.pickle"

    out_name = filename.split(".")[0] + ".html"
    html_writer = HtmlVisualizer(out_name, dark_mode=False)
    data = EstimatorPredictionViewerGosford(filename)
    tokenizer = data.tokenizer
    for inst_i, entry in enumerate(data):
        tokens = entry.get_mask_resolved_input_mask_with_input()
        highlight = lmap(is_mask, tokens)
        scores = entry.get_vector("overlap_score")
        pos_list = entry.get_vector("masked_lm_positions")
        probs = entry.get_vector("masked_lm_log_probs")
        probs = np.reshape(probs, [20, -1])
        rows = []
        for score, position, prob in zip(scores, pos_list, probs):
            tokens[position] = "{}-".format(position) + tokens[position]



            row = [Cell(position), Cell(score)]

            for idx in np.argsort(prob)[::-1][:5]:
                term = tokenizer.inv_vocab[idx]
                p = math.exp(prob[idx])
                row.append(Cell(term))
                row.append(Cell(p))
            rows.append(row)

        cells = data.cells_from_tokens(tokens, highlight)
        for score, position in zip(scores, pos_list):
            cells[position].highlight_score = score / 10000 * 255

        html_writer.multirow_print(cells, 20)
        html_writer.write_table(rows)


def view_grad_overlap_hidden():
    filename = "ukp_feature_overlap.pickle"
    obj = pickle.load(open(os.path.join(output_path, filename), "rb"))


    out_name = filename.split(".")[0] + ".html"
    html_writer = HtmlVisualizer(out_name, dark_mode=False)
    data = EstimatorPredictionViewerGosford(filename)

    for inst_i, entry in enumerate(data):
        tokens = entry.get_mask_resolved_input_mask_with_input()
        h_overlap = entry.get_vector('h_overlap')

        std = np.std(h_overlap, axis=2)
        mean = np.mean(h_overlap, axis=2)
        h_overlap = np.sum(h_overlap, axis=2)

        highlight = lmap(is_mask, tokens)
        cells = data.cells_from_tokens(tokens, highlight)
        rows = [cells]
        for layer_i in range(12):
            e = h_overlap[layer_i, :]
            e = [v * 1e6 for v in e]
            cells = data.cells_from_scores(e)
            rows.append(cells)

            e = [v * 1e8 for v in std[layer_i, :]]
            cells2 = data.cells_from_scores(e)
            rows.append(cells2)

        print(entry.get_vector("masked_lm_example_loss"))
        html_writer.multirow_print_from_cells_list(rows, 40)


def view_grad_overlap():
    filename = "gradient_overlap_4K.pickle"

    out_name = filename.split(".")[0] + ".html"
    html_writer = HtmlVisualizer(out_name, dark_mode=False)

    data = EstimatorPredictionViewerGosford(filename)
    iba = IntBinAverage()
    scores = []
    for inst_i, entry in enumerate(data):
        masked_lm_example_loss = entry.get_vector("masked_lm_example_loss")
        score = entry.get_vector("overlap_score")

        if masked_lm_example_loss > 1:
            norm_score = score / masked_lm_example_loss
            iba.add(masked_lm_example_loss, norm_score)
        scores.append(score)

    score_avg = average(scores)
    score_std = np.std(scores)

    avg = iba.all_average()
    std_dict = {}
    for key, values in iba.list_dict.items():
        std_dict[key] = np.std(values)
        if len(values) == 1:
            std_dict[key] = 999

    def unlikeliness(value, mean, std):
        return abs(value - mean) / std

    data = EstimatorPredictionViewerGosford(filename)
    print("num record : ", data.data_len)
    cnt = 0
    for inst_i, entry in enumerate(data):
        tokens = entry.get_mask_resolved_input_mask_with_input()
        masked_lm_example_loss = entry.get_vector("masked_lm_example_loss")
        highlight = lmap(is_mask, tokens)
        score = entry.get_vector("overlap_score")
        print(score)
        cells = data.cells_from_tokens(tokens, highlight)
        if masked_lm_example_loss > 1:
            bin_key = int(masked_lm_example_loss)
            norm_score = score / masked_lm_example_loss
            if norm_score > 5000:
                cnt += 1
            expectation = avg[bin_key]
            if unlikeliness(score, score_avg, score_std) > 2 or True:
                html_writer.multirow_print(cells, 20)
                if norm_score > expectation:
                    html_writer.write_paragraph("High")
                else:
                    html_writer.write_paragraph("Low")
                html_writer.write_paragraph("Norm score: " + str(norm_score))
                html_writer.write_paragraph("score: " + str(score))
                html_writer.write_paragraph("masked_lm_example_loss: " + str(masked_lm_example_loss))
                html_writer.write_paragraph("expectation: " + str(expectation))
    print("number over 5000: ", cnt)


if __name__ == '__main__':
    view_grad_overlap_hidden()