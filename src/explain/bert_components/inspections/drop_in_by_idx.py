import random

import numpy as np
import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.bert_components.cls_probe_w_visualize import load_probe, load_data, execute_with_attention_masking, \
    write_html
from explain.bert_components.cmd_nli import ModelConfig
from visualize.html_visual import HtmlVisualizer


def attention_modifier_by_drop_incoming(drop_src, drop_target, drop_layer):

    def get_modified_attention_mask(attention_mask, layer_no):
        if layer_no != drop_layer:
            return attention_mask
        else:
            attention_mask_np = np.array(attention_mask)
            # for j in range(len(attention_mask_np)):
            #     attention_mask_np[0, j, drop_idx] = 0
            seq_length = attention_mask_np.shape[1]
            attention_mask_np[0, drop_src, drop_target] = 0
            modified_attention_mask = tf.constant(attention_mask_np)
            return modified_attention_mask

    return get_modified_attention_mask


def attention_modifier_by_drop_incoming_list(drop_src, drop_target_list, drop_layer):
    def get_modified_attention_mask(attention_mask, layer_no):
        if layer_no != drop_layer:
            return attention_mask
        else:
            attention_mask_np = np.array(attention_mask)
            # for j in range(len(attention_mask_np)):
            #     attention_mask_np[0, j, drop_idx] = 0
            seq_length = attention_mask_np.shape[1]
            for drop_target in drop_target_list:
                attention_mask_np[0, drop_src, drop_target] = 0
            modified_attention_mask = tf.constant(attention_mask_np)
            return modified_attention_mask

    return get_modified_attention_mask



def main2():
    model_config = ModelConfig()
    model, bert_cls_probe = load_probe(model_config)
    batches = load_data(model_config)
    x0, x1, x2, y = batches[0]
    X = (x0, x1, x2)
    drop_condition_list = [range(15, 31), range(21, 31), range(15, 29),
                           [16, 17, 18, 20, 21, 22, 29, 30],
                           [16, 17, 20, 21, 22, 29],
                           [1, 2, 3, 4, 15, 16, 17, 18, 19, 21, 22, 29]
                           ]
    last_condition = [22]
    def enum_drop_conditions():
        repeat = True
        while repeat:
            k = random.randint(1, 31)
            if k not in last_condition:
                last_condition.append(k)
                yield last_condition

            if len(last_condition) >= 31:
                repeat = False

    tokenizer = get_tokenizer()
    target_layer = 9
    drop_idx = 22

    save_name = "cls_probe_drop_in_by_idx2.html"
    html = HtmlVisualizer(save_name)
    for drop_target_list in enum_drop_conditions():
        get_modified_attention_mask = attention_modifier_by_drop_incoming_list(drop_idx, drop_target_list,
                                                                               target_layer)
        input_ids = x0
        input_ids_slice = [input_ids[0][k] for k in drop_target_list]
        dropped_token = tokenizer.convert_ids_to_tokens(input_ids_slice)
        dropped_token_s = ",".join(dropped_token)
        drop_target_s = ",".join(map(str, drop_target_list))
        print("Attention dropped tokens:", dropped_token_s)
        messages = ["Drop {}-th token ({}) at layer_{}".format(
            drop_target_s, dropped_token_s, target_layer)]
        logits, probes = execute_with_attention_masking(bert_cls_probe, X, y, get_modified_attention_mask)

        def highlight_term(layer_no_plus_one, seq_idx):
            layer_no = layer_no_plus_one - 1
            # This 'layer_no' includes embedding layer
            if target_layer == layer_no and seq_idx in drop_target_list:
                return " (X)"
            else:
                return ""

        write_html(html, x0, logits, probes, y, messages, highlight_term)


def main():
    model_config = ModelConfig()
    model, bert_cls_probe = load_probe(model_config)
    batches = load_data(model_config)
    x0, x1, x2, y = batches[0]
    X = (x0, x1, x2)
    drop_condition_list = range(1, 30)
    tokenizer = get_tokenizer()
    target_layer = 9
    drop_idx = 22

    save_name = "cls_probe_drop_in_by_idx.html"
    html = HtmlVisualizer(save_name)
    for drop_target in drop_condition_list:
        get_modified_attention_mask = attention_modifier_by_drop_incoming(drop_idx, drop_target, target_layer)
        input_ids = x0
        dropped_token = tokenizer.convert_ids_to_tokens(input_ids[:, drop_target])
        print("Attention dropped tokens:", dropped_token[0])
        messages = ["Drop {}-th token ({}) at layer_{}".format(
            drop_target, dropped_token[0], target_layer)]
        logits, probes = execute_with_attention_masking(bert_cls_probe, X, y, get_modified_attention_mask)
        def highlight_term(layer_no_plus_one, seq_idx):
            layer_no = layer_no_plus_one - 1
            # This 'layer_no' includes embedding layer
            if target_layer == layer_no and drop_idx == seq_idx:
                return " (X)"
            else:
                return ""

        write_html(html, x0, logits, probes, y, messages, highlight_term)


if __name__ == "__main__":
    main2()


