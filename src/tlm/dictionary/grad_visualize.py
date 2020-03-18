import os
import pickle

import numpy as np

from cpath import output_path, data_path
from data_generator import tokenizer_wo_tf
from list_lib import right
from visualize.html_visual import get_color, Cell, HtmlVisualizer


def load_and_analyze_gradient():
    p = os.path.join(output_path, "dict_grad1.pickle")
    data = pickle.load(open(p, "rb"))
    data = data[0]
    tokenizer = tokenizer_wo_tf.FullTokenizer(os.path.join(data_path, "bert_voca.txt"))

    analyze_gradient(data, tokenizer)

def parse_data(data):
    data = data[0]
    gradients = data['gradients']
    d_input_ids = data['d_input_ids']
    input_ids = data['masked_input_ids']
    masked_lm_positions = data["masked_lm_positions"]
    return input_ids, d_input_ids, gradients, masked_lm_positions


def reshape_gradienet(gradients, n_inst, def_len, hidden_dim):
    r = np.reshape(gradients, [-1, n_inst, def_len, hidden_dim])
    return np.transpose(r, [1,0,2,3])


def analyze_gradient(data, tokenizer):
    gradients = data['gradients']
    d_input_ids = data['d_input_ids']
    mask_input_ids = data['masked_input_ids']
    masked_lm_positions = data["masked_lm_positions"]

    n_inst, seq_len = mask_input_ids.shape
    n_inst2, def_len = d_input_ids.shape

    assert n_inst == n_inst2

    def_len = 256
    hidden_dim = 768
    reshaped_grad = reshape_gradienet(gradients, n_inst, def_len, hidden_dim)
    print(reshaped_grad.shape)

    n_pred = reshaped_grad.shape[1]

    grad_per_token = np.sum(np.abs(reshaped_grad), axis=3)

    html_writer = HtmlVisualizer("dict_grad.html", dark_mode=False)


    for inst_idx in range(n_inst):
        tokens = tokenizer.convert_ids_to_tokens(mask_input_ids[inst_idx])
        #ans_tokens = tokenizer.convert_ids_to_tokens(input_ids[inst_idx])
        for i in range(len(tokens)):
            if tokens[i] == "[MASK]":
                tokens[i] = "[MASK_{}]".format(i)
            if tokens[i] == "[SEP]":
                tokens[i] = "[SEP]<br>"
        def_tokens = tokenizer.convert_ids_to_tokens(d_input_ids[inst_idx])
        s = tokenizer_wo_tf.pretty_tokens(tokens)

        lines = []

        grad_total_max = 0
        for pred_idx in range(n_pred):
            row = []
            max_val = max(grad_per_token[inst_idx, pred_idx])
            total = sum(grad_per_token[inst_idx, pred_idx])
            mask_pos = masked_lm_positions[inst_idx, pred_idx]

            if total > grad_total_max:
                grad_total_max = total


            row.append(Cell(mask_pos))
            row.append(Cell(int(total)))

            for def_idx in range(def_len):
                term = def_tokens[def_idx]
                cont_right = def_idx +1 < def_len and def_tokens[def_idx][:2] == "##"
                cont_left = term[:2] == "##"

                space_left = "&nbsp;" if not cont_left else ""
                space_right = "&nbsp;" if not cont_right else ""

                if term == "[PAD]":
                    break
                if term == "[unused5]":
                    term = "[\\n]"

                score = grad_per_token[inst_idx, pred_idx, def_idx]/(hidden_dim*2)
                bg_color = get_color(score)

                row.append(Cell(term, score, not cont_left, not cont_right))
                print("{}({})".format(term, grad_per_token[inst_idx, pred_idx, def_idx]), end=" ")

            lines.append((mask_pos, row))
            print("")
        lines.sort(key=lambda x:x[0])

        s = s.replace("[unused4]", "<b>DictTerm</b>")
        html_writer.write_paragraph(s)

        if grad_total_max > 5000000:
            html_writer.write_headline("HIGH Gradient")

        rows = right(lines)
        html_writer.write_table(rows)

        print("----------")
    html_writer.close()


if __name__ == '__main__':
    load_and_analyze_gradient()

