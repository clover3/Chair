import os
import sys

import numpy as np

from data_generator.bert_input_splitter import split_p_h_with_input_ids
from models.classic.stopword import load_stopwords_for_query
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from visualize.html_visual import HtmlVisualizer, Cell


def main():
    file_path = sys.argv[1]
    name = os.path.basename(file_path)
    viewer = EstimatorPredictionViewer(file_path)
    html = HtmlVisualizer("toke_score_gold.html")
    stopwords = load_stopwords_for_query()

    skip = 10
    for entry_idx, entry in enumerate(viewer):
        if entry_idx % skip != 0:
            continue
        tokens = entry.get_tokens("input_ids")
        input_ids = entry.get_vector("input_ids")
        label_ids = entry.get_vector("label_ids")
        label_ids = np.reshape(label_ids, [-1, 2])
        log_label_ids = np.log(label_ids + 1e-10)
        seg1, seg2 = split_p_h_with_input_ids(tokens, input_ids)

        pad_idx = tokens.index("[PAD]")
        assert pad_idx > 0

        logits = entry.get_vector("logits")
        cells = []
        cells2 = []
        for idx in range(pad_idx):
            probs = label_ids[idx]
            token = tokens[idx]

            score = probs[0]
            color = "B" if score > 0 else "R"
            highlight_score = min(abs(score) * 10000, 100)
            if token in stopwords:
                highlight_score = 0
            if token in seg1:
                highlight_score = 50
                color = "G"

            c = Cell(token, highlight_score=highlight_score, target_color=color)
            cells.append(c)
        html.multirow_print_from_cells_list([cells, cells2])


        if entry_idx > 10000:
            break



if __name__ == "__main__":
    main()