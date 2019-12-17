import os
import pickle

import math
import numpy as np

from data_generator.common import get_tokenizer
from path import output_path
from visualize.html_visual import Cell, HtmlVisualizer


def visual(filename):
    tokenizer = get_tokenizer()
    p = os.path.join(output_path, filename)
    data = pickle.load(open(p, "rb"))

    batch_size, seq_length = data[0]['input_ids'].shape

    keys = list(data[0].keys())
    vectors = {}

    for e in data:
        for key in keys:
            if key not in vectors:
                vectors[key] = []
            vectors[key].append(e[key])

    for key in keys:
        vectors[key] = np.concatenate(vectors[key], axis=0)

    any_key = keys[0]
    data_len = len(vectors[any_key])
    html = HtmlVisualizer("all_losses.html")
    num_predictions = len(vectors["grouped_positions"][0][0])
    for i in range(data_len):
        input_ids = vectors["input_ids"][i]

        mask_valid = [0] * seq_length
        loss1_arr = [-2] * seq_length
        loss2_arr = [-2] * seq_length
        positions = vectors["grouped_positions"][i]
        num_trials = len(positions)
        for t_i in range(num_trials):
            print(vectors["grouped_positions"][i][t_i])
            for p_i in range(num_predictions):
                loc = vectors["grouped_positions"][i][t_i][p_i]
                loss1 = vectors["grouped_loss1"][i][t_i][p_i]
                loss2 = vectors["grouped_loss2"][i][t_i][p_i]

                loss1_arr[loc] = math.exp(-loss1)
                loss2_arr[loc] = math.exp(-loss2)
                assert mask_valid[loc] == 0
                mask_valid[loc] = 1
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        row0 = [Cell("Text:")]
        row1 = [Cell("BERT:")]
        row2 = [Cell("RFT :")]
        for i, t in enumerate(tokens):
            if t == '[PAD]':
                break

            s = 0
            drop = loss1_arr[i] - loss2_arr[i]
            s = drop * 100
            color = "B"
            if s < 0:
                color = "R"
                s = -s
            row0.append(Cell(t, target_color=color))
            row1.append(Cell(loss1_arr[i], s, target_color=color))
            row2.append(Cell(loss2_arr[i], s, target_color=color))
        html.write_table([row0, row1, row2])



if __name__ == "__main__":
    visual("{}_{}.pickle")