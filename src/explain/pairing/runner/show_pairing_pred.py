from cache import load_from_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from misc_lib import two_digit_float
from trainer.np_modules import sigmoid
from visualize.html_visual import HtmlVisualizer, Cell


def main():
    output_d = load_from_pickle("pairing_pred")
    html = HtmlVisualizer("pairing.html")
    tokenizer = get_tokenizer()
    logits_grouped_by_layer = output_d["logits"]
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
        ex_scores = get('ex_scores')
        html.write_paragraph("label={}".format(get("label")))

        row1 = [Cell("")] + list([Cell(t, int(s*100)) for t, s in zip(tokens, ex_scores)])
        row2 = float_arr_to_cell("ex_prob", ex_scores)
        for i, s in enumerate(ex_scores):
            if s > 0.5:
                row2[i+1].highlight_score = 100

        rows = [row1, row2]

        for layer_no in range(num_layers):
            layer_logit = logits_grouped_by_layer[layer_no][data_idx]
            probs = sigmoid(layer_logit)
            row = float_arr_to_cell("layer_{}".format(layer_no), probs[:, 1])
            rows.append(row)

        html.write_table(rows)



if __name__ == "__main__":
    main()