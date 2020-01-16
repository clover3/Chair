import numpy as np

from misc_lib import lmap, IntBinAverage
from tlm.estimator_prediction_viewer import EstimatorPredictionViewerGosford
from tlm.token_utils import is_mask
from visualize.html_visual import HtmlVisualizer


def loss_view():
    filename2 = "ukp_sample.pickle"

    out_name = filename2.split(".")[0] + ".html"
    html_writer = HtmlVisualizer(out_name, dark_mode=False)

    filename = "ukp_lm_overlap.pickle"
    data = EstimatorPredictionViewerGosford(filename)
    iba = IntBinAverage()
    for inst_i, entry in enumerate(data):
        masked_lm_example_loss = entry.get_vector("masked_lm_example_loss")
        score = entry.get_vector("overlap_score")

        if masked_lm_example_loss > 1:
            norm_score = score / masked_lm_example_loss
            iba.add(masked_lm_example_loss, norm_score)


    avg = iba.all_average()
    std_dict = {}
    for key, values in iba.list_dict.items():
        std_dict[key] = np.std(values)
        if len(values) == 1:
            std_dict[key] = 999



    def unlikeliness(value, mean, std):
        return abs(value - mean) / std

    data = EstimatorPredictionViewerGosford(filename2)
    for inst_i, entry in enumerate(data):
        tokens = entry.get_mask_resolved_input_mask_with_input()
        masked_lm_example_loss = entry.get_vector("masked_lm_example_loss")
        highlight = lmap(is_mask, tokens)
        score = entry.get_vector("overlap_score")
        cells = data.cells_from_tokens(tokens, highlight)
        if masked_lm_example_loss > 1:
            bin_key = int(masked_lm_example_loss)
            norm_score = score / masked_lm_example_loss
            expectation = avg[bin_key]
            if unlikeliness(norm_score, expectation, std_dict[bin_key]) > 2 or True:
                html_writer.multirow_print(cells, 20)
                if norm_score > expectation:
                    html_writer.write_paragraph("High")
                else:
                    html_writer.write_paragraph("Low")
                html_writer.write_paragraph("Norm score: " + str(norm_score))
                html_writer.write_paragraph("score: " + str(score))
                html_writer.write_paragraph("masked_lm_example_loss: " + str(masked_lm_example_loss))
                html_writer.write_paragraph("expectation: " + str(expectation))

if __name__ == '__main__':
    loss_view()