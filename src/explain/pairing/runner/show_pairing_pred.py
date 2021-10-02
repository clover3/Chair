import scipy.special

from cache import load_from_pickle
from contradiction.medical_claims.token_tagging.deletion_score_to_html import make_prediction_summary_str
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from misc_lib import two_digit_float
from trainer.np_modules import sigmoid
from visualize.html_visual import HtmlVisualizer, Cell


def main():
    save_name = "alamri_mismatch_all"
    output_d = load_from_pickle(save_name)
    html = HtmlVisualizer("alamri_mismatch.html")
    tokenizer = get_tokenizer()
    logits_grouped_by_layer = output_d["per_layer_logits"]
    num_layers = 12

    def float_arr_to_cell(head, float_arr):
        return [Cell(head)] + lmap(Cell, map(two_digit_float, float_arr))

    def float_arr_to_cell2(head, float_arr):
        return [Cell(head)] + lmap(Cell, map("{0:.4f}".format, float_arr))

    num_data = len(output_d['input_ids'])
    for data_idx in range(num_data)[:100]:
        def get(name):
            return output_d[name][data_idx]

        tokens = tokenizer.convert_ids_to_tokens(get("input_ids"))
        first_padding_loc = tokens.index("[PAD]")
        display_len = first_padding_loc + 1
        ex_scores = get('ex_scores')
        probs = scipy.special.softmax(get('logits'))

        pred_str = make_prediction_summary_str(probs)

        html.write_paragraph("Prediction: {}".format(pred_str))
        html.write_paragraph("gold label={}".format(get("label")))

        text_rows = [Cell("")] + list([Cell(t, int(s*100)) for t, s in zip(tokens, ex_scores)])
        text_rows = text_rows
        ex_probs = float_arr_to_cell("ex_prob", ex_scores)
        for i, s in enumerate(ex_scores):
            if s > 0.5:
                ex_probs[i+1].highlight_score = 100

        rows = [text_rows, ex_probs]

        mid_pred_rows = []
        for layer_no in range(num_layers):
            layer_logit = logits_grouped_by_layer[layer_no][data_idx]
            probs = sigmoid(layer_logit)
            row = float_arr_to_cell("layer_{}".format(layer_no), probs[:, 1])
            mid_pred_rows.append(row)

        rows.extend(mid_pred_rows[::-1])

        rows = [row[:display_len] for row in rows]
        html.write_table(rows)


if __name__ == "__main__":
    main()