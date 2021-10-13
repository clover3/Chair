from typing import List

import scipy.special

from cache import load_from_pickle
from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_prediction_summary_str
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import HtmlVisualizer, Cell


def print_html(score_pickle_name, save_name):
    output_d = load_from_pickle(score_pickle_name)
    html = HtmlVisualizer(save_name)
    tokenizer = get_tokenizer()
    logits_grouped_by_layer = output_d["per_layer_logits"]
    num_layers = 12 + 1
    def layer_no_to_name(layer_no):
        if layer_no == 0:
            return "embed"
        else:
            return "layer_{}".format(layer_no-1)

    num_data = len(output_d['input_ids'])
    for data_idx in range(num_data)[:100]:
        def get(name):
            try:
                return output_d[name][data_idx]
            except KeyError:
                if name == "label":
                    return 0
                else:
                    raise

        tokens = tokenizer.convert_ids_to_tokens(get("input_ids"))
        first_padding_loc = tokens.index("[PAD]")
        display_len = first_padding_loc + 1
        probs = scipy.special.softmax(get('logits'))

        pred_str = make_prediction_summary_str(probs)

        html.write_paragraph("Prediction: {}".format(pred_str))
        html.write_paragraph("gold label={}".format(get("label")))

        text_rows = [Cell("")] + cells_from_tokens(tokens)
        rows = [text_rows]

        mid_pred_rows = []
        for layer_no in range(num_layers):
            layer_logit = logits_grouped_by_layer[layer_no][data_idx]
            probs = scipy.special.softmax(layer_logit, axis=1)
            def prob_to_one_digit(p):
                v = int(p * 10 + 0.05)
                if v > 9:
                    return "A"
                else:
                    s = str(v)
                    assert len(s) == 1
                    return s

            row = [Cell(layer_no_to_name(layer_no))]
            for seq_idx in range(len(probs)):
                case_probs = probs[seq_idx]
                prob_digits: List[str] = list(map(prob_to_one_digit, case_probs))
                cell_str = "".join(prob_digits)
                color_mapping = {
                    0: 2,  # Red = Contradiction
                    1: 1,  # Green = Neutral
                    2: 0   # Blue = Entailment
                }

                color_score = [255 * case_probs[color_mapping[i]] for i in range(3)]
                color = "".join([("%02x" % int(v)) for v in color_score])
                cell = Cell(cell_str, 255, target_color=color)
                row.append(cell)

            mid_pred_rows.append(row)

        rows.extend(mid_pred_rows[::-1])

        rows = [row[:display_len] for row in rows]
        html.write_table(rows)


def main():
    score_pickle_name = "nli_probe_gove_site"
    # score_pickle_name = "nli_probe_gove_site_middle_in"
    # score_pickle_name = "nli_probe_replace_layer10_op2"
    # score_pickle_name = "nli_probe_replace_layer11_op2"
    save_name = score_pickle_name + ".html"
    print_html(score_pickle_name, save_name)


if __name__ == "__main__":
    main()