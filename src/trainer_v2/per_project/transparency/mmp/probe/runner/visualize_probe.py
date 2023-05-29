import sys

from cpath import output_path
from explain.pairing.run_visualizer.show_cls_probe import MMDVisualize, RegressionVisualize
from misc_lib import path_join
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
from trainer_v2.custom_loop.dataset_factories import get_pairwise_dataset
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser
from visualize.html_visual import HtmlVisualizer, Cell


def print_html(output_d, save_name):
    html = HtmlVisualizer(save_name)
    tokenizer = get_tokenizer()
    logits_grouped_by_layer = output_d["per_layer_logits"]
    # shape: [num_layer, dat_idx, seq_idx, label_idx
    rel_probs = logits_grouped_by_layer[:, :, :, 0]
    avg_p = np.mean(rel_probs)
    per_layer_mean = np.mean(rel_probs, axis=2)
    per_layer_mean = np.mean(per_layer_mean, axis=1)

    print("Probe mean", avg_p)
    print("Per layer mean", per_layer_mean)
    visualize_policy = RegressionVisualize(avg_p)
    num_layers = len(logits_grouped_by_layer)
    def layer_no_to_name(layer_no):
        if layer_no == 0:
            return "embed"
        elif layer_no < 4:
            return "layer_{}".format(layer_no-1)
        else:
            skip = 6
            return "layer_{}".format(layer_no + skip - 1)

    num_data = len(output_d['input_mask'])
    print("Num data", num_data)
    num_print = min(100, num_data)

    for data_idx in range(num_print):
        def get(name):
            try:
                return output_d[name][data_idx]
            except KeyError:
                if name == "label":
                    return 0
                else:
                    raise

        input_ids = get("input_ids")
        input_mask = get("input_mask")

        for token_id, mask_val in zip(input_ids, input_mask):
            is_padding = (token_id == 0)
            if is_padding and bool(mask_val):
                print(input_ids)
                print(input_mask)
                print(token_id)
                print(mask_val)
                raise Exception()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        try:
            first_padding_loc = tokens.index("[PAD]")
            display_len = first_padding_loc + 1
        except ValueError:
            display_len = len(tokens)

        logits = get('logits')
        pred_str = "{0:.2f}".format(logits[0])
        html.write_paragraph("Prediction: {}".format(pred_str))
        html.write_paragraph("gold label={}".format(get("label")))

        text_rows = [Cell("")] + cells_from_tokens(tokens)
        rows = [text_rows]

        mid_pred_rows = []
        for layer_no in range(num_layers):
            layer_logit = logits_grouped_by_layer[layer_no][data_idx]
            row = [Cell(layer_no_to_name(layer_no))]
            for seq_idx in range(len(layer_logit)):
                # rel = layer_logit[seq_idx][0] - per_layer_mean[layer_no]
                rel = layer_logit[seq_idx][0]
                cell_str = visualize_policy.get_cell_str(rel)
                rel_w = rel * 10
                color_score = visualize_policy.prob_to_color(rel_w)
                color = "".join([("%02x" % int(v)) for v in color_score])
                cell = Cell(cell_str, 255, target_color=color)
                row.append(cell)

            mid_pred_rows.append(row)

        rows.extend(mid_pred_rows[::-1])
        rows = [row[:display_len] for row in rows]
        html.write_table(rows)


def re_order(doubled_pred: np.array, batch_size):
    seen_batch_size = batch_size * 2
    cursor = 0
    probe_out1 = []
    probe_out2 = []
    while cursor < len(doubled_pred):
        num_remain = len(doubled_pred) - cursor
        if num_remain >= seen_batch_size:
            cur_seen_batch_size = seen_batch_size
        else:
            cur_seen_batch_size = num_remain

        cur_batch_size = int(cur_seen_batch_size / 2)
        batch = doubled_pred[cursor: cursor + cur_seen_batch_size]
        a = batch[:cur_batch_size]
        b = batch[cur_batch_size:]
        probe_out1.append(a)
        probe_out2.append(b)
        cursor += cur_seen_batch_size

    return np.concatenate(probe_out1, axis=0), np.concatenate(probe_out2, axis=0)


def re_order_and_stack_reshape(v, batch_size):
    pos, neg = re_order(v, batch_size)
    all = []
    for i in range(len(pos)):
        all.append(pos[i])
        all.append(neg[i])
    return np.stack(all, axis=0)

    # t = np.stack([pos, neg], axis=1)
    # B, _, M, _ = t.shape
    # return np.reshape(t, [B*2, M, -1])


def reform_logits(v, batch_size):
    pos, neg = re_order(v, batch_size)
    all = []
    for i in range(len(pos)):
        all.append(pos[i])
        all.append(neg[i])
    return np.stack(all, axis=0)

    # t = np.stack([pos, neg], axis=1)
    # B, _, _ = t.shape
    # return np.reshape(t, [B*2, -1])


def get_input_ids(args):
    run_config: RunConfig2 = get_run_config_for_predict(args)

    def build_dataset(input_files, is_for_training):
        return get_pairwise_dataset(
            input_files, run_config, ModelConfig256_1(), is_for_training, add_dummy_y=False)

    dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
    dataset = dataset.take(3)

    pos = []
    neg = []
    for item in iter(dataset):
        pos.append(item[f'input_ids1'])
        neg.append(item[f'input_ids2'])

    pos_neg = [np.concatenate(pos, axis=0), np.concatenate(neg, axis=0)]
    t = np.stack(pos_neg, axis=1)
    B, _, M = t.shape
    return np.reshape(t, [B * 2, M])


def main(args):
    run_name = "tp7"
    input_ids = get_input_ids(args)
    n_item = len(input_ids)
    n_pair = int(n_item / 2)

    labels = []
    for _ in range(n_pair):
        labels.append(1)
        labels.append(0)

    output_d = load_from_pickle("probe_inf_dev")
    batch_size = 16
    probe_d = {}
    for key in output_d['probe_on_hidden']:
        v = output_d['probe_on_hidden'][key]
        probe_d[key] = re_order_and_stack_reshape(v, batch_size)

    probe_stack = []
    for layer_no in [0, 1, 2, 3, 10, 11, 12]:
        key = f"layer_{layer_no}"
        probe_stack.append(probe_d[key])

    input_mask = re_order_and_stack_reshape(output_d['input_mask'], batch_size)
    probe = np.stack(probe_stack, axis=0)
    save_name = run_name + ".html"

    new_output_d = {
        "per_layer_logits": probe,
        "logits": reform_logits(output_d['logits'], batch_size),
        "input_ids": input_ids,
        "label": labels,
        "input_mask": input_mask,
    }
    print_html(new_output_d, save_name)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)



