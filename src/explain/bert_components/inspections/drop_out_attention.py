import numpy as np
import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.bert_components.cls_probe_w_visualize import load_probe, load_data, execute_with_attention_masking, \
    write_html
from explain.bert_components.cmd_nli import ModelConfig
from visualize.html_visual import HtmlVisualizer


def attention_modifier_by_drop_out_going(drop_idx, drop_after):
    def get_modified_attention_mask(attention_mask, layer_no):
        if layer_no < drop_after:
            return attention_mask
        else:
            attention_mask_np = np.array(attention_mask)
            # for j in range(len(attention_mask_np)):
            #     attention_mask_np[0, j, drop_idx] = 0
            attention_mask_np[0, 0, drop_idx] = 0
            print(attention_mask_np)
            modified_attention_mask = tf.constant(attention_mask_np)
            return modified_attention_mask

    return get_modified_attention_mask


def main():
    model_config = ModelConfig()
    model, bert_cls_probe = load_probe(model_config)
    # bert_cls_probe, dev_insts = load_cls_probe()
    # batches = get_batches_ex(dev_insts, 32, 4)
    batches = load_data(model_config)
    x0, x1, x2, y = batches[0]
    X = (x0, x1, x2)
    # drop_condition_list = [(9, 0), (9, 22), (9, 25), (10, 22), (11, 22), (12, 22)]
    # drop_layer = 7
    # drop_condition_list = [(drop_layer, k) for k in range(1, 50)]
    drop_condition_list = [(9, 22), (10, 22), (11, 22), (12, 22)]
    # drop_condition_list = [(j, 29) for j in range(12)]
    tokenizer = get_tokenizer()

    save_name = "cls_probe_drop_diff_layer.html"
    html = HtmlVisualizer(save_name)
    for cut_layer, drop_idx in drop_condition_list:
        get_modified_attention_mask = attention_modifier_by_drop_out_going(drop_idx, cut_layer)

        input_ids = x0
        dropped_token = tokenizer.convert_ids_to_tokens(input_ids[:, drop_idx])
        print("Attention dropped tokens:", dropped_token[0])
        messages = ["Drop {}-th token ({}) starting from layer_{}".format(
            drop_idx, dropped_token[0], cut_layer)]
        logits, probes = execute_with_attention_masking(bert_cls_probe, X, y, get_modified_attention_mask)
        def highlight_term(layer_no, seq_idx):
            # This 'layer_no' includes embedding layer
            if cut_layer <= layer_no - 1 and drop_idx == seq_idx:
                return " (X)"
            else:
                return ""

        write_html(html, x0, logits, probes, y, messages, highlight_term)


if __name__ == "__main__":
    main()


