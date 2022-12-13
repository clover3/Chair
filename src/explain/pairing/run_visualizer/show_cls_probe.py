import abc
import os
from typing import List

import numpy as np
import scipy.special

from cache import load_from_pickle
from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_nli_prediction_summary_str
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.relevance_text import make_relevance_prediction_summary_str
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import HtmlVisualizer, Cell


class TaskVisualizationPolicyI(abc.ABC):
    @abc.abstractmethod
    def make_prediction_summary_str(self, base_prob):
        return make_nli_prediction_summary_str(base_prob)

    @abc.abstractmethod
    def prob_to_color(self, prob):
        color_mapping = {
            0: 2,  # Red = Contradiction
            1: 1,  # Green = Neutral
            2: 0  # Blue = Entailment
        }
        color_score = [255 * prob[color_mapping[i]] for i in range(3)]
        return color_score

    @abc.abstractmethod
    def get_cell_str(self, prob):
        pass


class NLIVisualize(TaskVisualizationPolicyI):
    @classmethod
    def make_prediction_summary_str(self, base_prob):
        return make_nli_prediction_summary_str(base_prob)

    @classmethod
    def prob_to_color(self, prob) -> List[int]:
        color_mapping = {
            0: 2,  # Red = Contradiction
            1: 1,  # Green = Neutral
            2: 0  # Blue = Entailment
        }
        color_score = [255 * prob[color_mapping[i]] for i in range(3)]
        return color_score

    @classmethod
    def get_cell_str(self, prob):
        def prob_to_one_digit(p):
            v = int(p * 10 + 0.05)
            if v > 9:
                return "A"
            else:
                s = str(v)
                assert len(s) == 1
                return s

        prob_digits: List[str] = list(map(prob_to_one_digit, prob))
        cell_str = "".join(prob_digits)
        return cell_str


class MMDVisualize(TaskVisualizationPolicyI):
    def __init__(self, base_p):
        self.base_p = base_p

    def make_prediction_summary_str(self, base_prob):
        return make_relevance_prediction_summary_str(base_prob)

    def prob_to_color(self, rel):
        color_mapping = {
            0: 0,  # Red = Contradiction
            1: 2,  # Green = Neutral
            2: 1  # Blue = Entailment
        }
        mag = min(int(abs(rel) * 255 / 10), 255)
        mag = 255 - mag
        if rel > 0:
            return [mag, mag, 255]
        else:
            return [255, mag, mag]

        color_score = [prob[color_mapping[0]], prob[color_mapping[0]], 1]
        color_score = [255 * v for v in color_score]
        return color_score

    def get_cell_str(self, rel):
        return "{0:.1f}".format(rel)
        s = "{0:.2f}".format(prob[1])
        if s[0] == 1:
            return "1.0"
        else:
            return s[1:]


def print_html_nli(output_d, visualize_policy: TaskVisualizationPolicyI, save_name):
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
    n_rel = 0
    n_non_rel = 0
    for data_idx in range(num_data):
        if n_rel == 200:
            break
        def get(name):
            try:
                return output_d[name][data_idx]
            except KeyError:
                if name == "label":
                    return 0
                else:
                    raise

        tokens = tokenizer.convert_ids_to_tokens(get("input_ids"))
        try:
            first_padding_loc = tokens.index("[PAD]")
            display_len = first_padding_loc + 1
        except ValueError:
            display_len = len(tokens)

        probs = scipy.special.softmax(get('logits'))
        if probs[1] > 0.5:
            n_rel += 1
        else:
            if n_non_rel > 5 * n_rel:
                continue
            else:
                n_non_rel += 1

        pred_str = visualize_policy.make_prediction_summary_str(probs)
        html.write_paragraph("Prediction: {}".format(pred_str))
        html.write_paragraph("gold label={}".format(get("label")))

        text_rows = [Cell("")] + cells_from_tokens(tokens)
        rows = [text_rows]

        mid_pred_rows = []
        for layer_no in range(num_layers):
            layer_logit = logits_grouped_by_layer[layer_no][data_idx]
            probs = scipy.special.softmax(layer_logit, axis=1)
            row = [Cell(layer_no_to_name(layer_no))]
            for seq_idx in range(len(probs)):
                case_probs = probs[seq_idx]
                cell_str = visualize_policy.get_cell_str(case_probs)
                color_score = visualize_policy.prob_to_color(case_probs)
                color = "".join([("%02x" % int(v)) for v in color_score])
                cell = Cell(cell_str, 255, target_color=color)
                row.append(cell)

            mid_pred_rows.append(row)

        rows.extend(mid_pred_rows[::-1])

        rows = [row[:display_len] for row in rows]
        html.write_table(rows)


def print_html(output_d, save_name):
    html = HtmlVisualizer(save_name)
    tokenizer = get_tokenizer()
    logits_grouped_by_layer = output_d["per_layer_logits"]
    #[num_layer, dat_idx, seq_idx, label_idx
    all_probs = scipy.special.softmax(logits_grouped_by_layer, axis=3)
    rel_probs = all_probs[:, :, :, 1]
    avg_p = np.mean(rel_probs)
    per_layer_mean = np.mean(rel_probs, axis=2)
    per_layer_mean = np.mean(per_layer_mean, axis=1)

    print("Probe mean", avg_p)
    print("Per layer mean", per_layer_mean)
    visualize_policy = MMDVisualize(avg_p)
    num_layers = 12 + 1
    def layer_no_to_name(layer_no):
        if layer_no == 0:
            return "embed"
        else:
            return "layer_{}".format(layer_no-1)

    num_data = len(output_d['input_ids'])
    n_rel = 0
    n_non_rel = 0
    for data_idx in range(num_data):
        if n_rel == 200:
            break
        def get(name):
            try:
                return output_d[name][data_idx]
            except KeyError:
                if name == "label":
                    return 0
                else:
                    raise

        tokens = tokenizer.convert_ids_to_tokens(get("input_ids"))
        try:
            first_padding_loc = tokens.index("[PAD]")
            display_len = first_padding_loc + 1
        except ValueError:
            display_len = len(tokens)

        probs = scipy.special.softmax(get('logits'))
        if probs[1] > 0.5:
            n_rel += 1
        else:
            if n_non_rel > 5 * n_rel:
                continue
            else:
                n_non_rel += 1

        pred_str = visualize_policy.make_prediction_summary_str(probs)
        html.write_paragraph("Prediction: {}".format(pred_str))
        html.write_paragraph("gold label={}".format(get("label")))

        text_rows = [Cell("")] + cells_from_tokens(tokens)
        rows = [text_rows]

        mid_pred_rows = []
        for layer_no in range(num_layers):
            layer_logit = logits_grouped_by_layer[layer_no][data_idx]
            probs = scipy.special.softmax(layer_logit, axis=1)
            row = [Cell(layer_no_to_name(layer_no))]
            for seq_idx in range(len(probs)):
                case_probs = probs[seq_idx]
                raw_p = case_probs[1]
                rel = np.log(raw_p / per_layer_mean[layer_no])
                cell_str = visualize_policy.get_cell_str(rel)
                color_score = visualize_policy.prob_to_color(rel)
                color = "".join([("%02x" % int(v)) for v in color_score])
                cell = Cell(cell_str, 255, target_color=color)
                row.append(cell)

            mid_pred_rows.append(row)

        rows.extend(mid_pred_rows[::-1])

        rows = [row[:display_len] for row in rows]
        html.write_table(rows)

def estimator_pickle_to_stacked(pickle_path):
    v = EstimatorPredictionViewer(pickle_path)
    per_logits = v.vectors['per_layer_logits']
    v.vectors['per_layer_logits'] = np.transpose(per_logits, [1, 0, 2, 3])
    return v.vectors


def main2():
    score_pickle_name = "nli_probe_gove_site"
    save_name = score_pickle_name + ".html"
    output_d = load_from_pickle(score_pickle_name)
    print_html_nli(output_d, NLIVisualize(), save_name)


def main():
    run_name = "mmd_Z_probe_B_50000"
    run_name = "mmd_Z_probe_D_20000"
    pickle_path = os.path.join(output_path, "align", run_name + ".score")
    v = estimator_pickle_to_stacked(pickle_path)
    save_name = run_name + ".html"
    print_html(v, save_name)


if __name__ == "__main__":
    main()
