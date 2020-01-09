import os
import pickle

import numpy as np

from cpath import output_path
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from visualize.html_visual import HtmlVisualizer, Cell


def per_doc_score():
    filename = "fetch_hidden_dim.pickle"
    html_writer = HtmlVisualizer("preserved.html", dark_mode=False)

    p = os.path.join(output_path, filename)
    raw_data = pickle.load(open(p, "rb"))


    n_skip = 0
    data = EstimatorPredictionViewer(filename)
    for inst_i, entry in enumerate(data):
        if inst_i > 100:
            break
        count_preserved = entry.get_vector("layer_count")
        tokens = entry.get_tokens("input_ids")
        cells = data.cells_from_tokens(tokens)
        valid_parst = count_preserved[:len(cells)]
        print(count_preserved.shape)
        return
        avg = np.average(count_preserved)
        row = []
        row2 = []
        f_print = avg > 20
        print(avg)
        if f_print:
            html_writer.write_paragraph("Skipped {} articles".format(n_skip))
            n_skip = 0
            for idx, cell in enumerate(cells):
                score = count_preserved[idx] / 728 * 100
                cell.highlight_score = score
                row.append(cell)
                row2.append((Cell(count_preserved[idx], score)))
                if len(row) == 20:
                    html_writer.write_table([row, row2])
                    row = []
                    row2 = []

            html_writer.write_paragraph(str(avg))
        else:
            n_skip += 1


if __name__ == '__main__':
    per_doc_score()