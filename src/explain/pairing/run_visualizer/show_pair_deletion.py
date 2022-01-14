from typing import List

from cache import load_from_pickle
from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_nli_prediction_summary_str
from cpath import at_output_dir
from explain.pairing.pair_deletion_common import summarize_pair_deletion_results, PerGroupSummary, load_p_h_pair_text
from list_lib import lmap, left, right
from misc_lib import two_digit_float
from visualize.html_visual import HtmlVisualizer, Cell, get_tooltip_cell


def main():
    save_name = "alamri_pair"
    info_entries, output_d = load_from_pickle(save_name)
    html = HtmlVisualizer("alamri_pairing_deletion.html", use_tooltip=True)
    initial_text = load_p_h_pair_text(at_output_dir("alamri_pilot", "true_pair_small.csv"))
    per_group_summary: List[PerGroupSummary] = summarize_pair_deletion_results(info_entries, output_d)

    def float_arr_to_str_arr(float_arr):
        return list(map(two_digit_float, float_arr))

    def float_arr_to_cell(head, float_arr):
        return [Cell(head)] + lmap(Cell, map(two_digit_float, float_arr))

    def float_arr_to_cell2(head, float_arr):
        return [Cell(head)] + lmap(Cell, map("{0:.4f}".format, float_arr))

    num_data = len(output_d['input_ids'])
    for data_idx, (p, h) in enumerate(initial_text):
        group_summary = per_group_summary[data_idx]

        p_tokens = p.split()
        h_tokens = h.split()

        base_score = group_summary.score_d[(-1, -1)]
        pred_str = make_nli_prediction_summary_str(base_score)
        html.write_paragraph("Prediction: {}".format(pred_str))

        keys = list(group_summary.score_d.keys())
        p_idx_max = max(left(keys))
        h_idx_max = max(right(keys))

        def get_pair_score_by_h(key):
            p_score, h_score = group_summary.effect_d[key]
            return h_score

        def get_pair_score_by_p(key):
            p_score, h_score = group_summary.effect_d[key]
            return p_score

        def get_table(get_pair_score_at):
            head = [Cell("")] + [Cell(t) for t in p_tokens]
            rows = [head]
            for h_idx in range(h_idx_max + 1):
                row = [Cell(h_tokens[h_idx])]
                for p_idx in range(p_idx_max + 1):
                    s = get_pair_score_at((p_idx, h_idx))
                    one_del_score = group_summary.score_d[(p_idx, -1)]
                    two_del_score = group_summary.score_d[(p_idx, h_idx)]
                    tooltip_str = "{} -> {}".format(float_arr_to_str_arr(one_del_score),
                                                    float_arr_to_str_arr(two_del_score))
                    row.append(get_tooltip_cell(two_digit_float(s), tooltip_str))
                rows.append(row)
            return rows

        html.write_table(get_table(get_pair_score_by_p))
        html.write_table(get_table(get_pair_score_by_h))
        html.write_bar()


if __name__ == "__main__":
    main()