from typing import List

import numpy as np
import scipy.special

from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_nli_prediction_summary_str
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import Cell


def prob_to_one_digit(p):
    v = int(p * 10 + 0.05)
    if v > 9:
        return "A"
    else:
        s = str(v)
        assert len(s) == 1
        return s


def layer_no_to_name(layer_no):
    if layer_no == 0:
        return "embed"
    else:
        return "layer_{}".format(layer_no-1)


def write_html(html, input_ids, logits, probe_logits, y):
    num_layers = 12 + 1
    print(len(probe_logits))
    print(probe_logits[0].shape)
    tokenizer = get_tokenizer()
    num_data = len(input_ids)
    probs_arr = scipy.special.softmax(logits, axis=-1)
    for data_idx in range(num_data)[:100]:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[data_idx])
        first_padding_loc = tokens.index("[PAD]")
        display_len = first_padding_loc + 1
        pred_str = make_nli_prediction_summary_str(probs_arr[data_idx])
        html.write_paragraph("Prediction: {}".format(pred_str))
        html.write_paragraph("gold label={}".format(y[data_idx]))
        text_rows = [Cell("")] + cells_from_tokens(tokens)
        rows = [text_rows]
        mid_pred_rows = []
        for layer_no in range(num_layers):
            layer_logit = probe_logits[layer_no][data_idx]
            probs = scipy.special.softmax(layer_logit, axis=1)
            head = Cell(layer_no_to_name(layer_no))
            row = get_row_cells(head, probs)
            mid_pred_rows.append(row)

        head = Cell("avg")
        hidden_layers_logits = np.array([probe_logits[i][data_idx] for i in range(1, 13)])
        print('hidden_layers_logits', hidden_layers_logits.shape)
        avg_probs = np.mean(scipy.special.softmax(hidden_layers_logits, axis=2), axis=0)
        print(avg_probs.shape)
        row = get_row_cells(head, avg_probs)
        mid_pred_rows.append(row)

        rows.extend(mid_pred_rows[::-1])

        rows = [row[:display_len] for row in rows]
        html.write_table(rows)


def get_row_cells(head, probs):
    row = [head]
    for seq_idx in range(len(probs)):
        case_probs = probs[seq_idx]
        prob_digits: List[str] = list(map(prob_to_one_digit, case_probs))
        cell_str = "".join(prob_digits)
        color_mapping = {
            0: 2,  # Red = Contradiction
            1: 1,  # Green = Neutral
            2: 0  # Blue = Entailment
        }
        color_score = [255 * case_probs[color_mapping[i]] for i in range(3)]
        color = "".join([("%02x" % int(v)) for v in color_score])
        cell = Cell(cell_str, 255, target_color=color)
        row.append(cell)
    return row