import numpy as np
import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.bert_components.cls_probe_w_visualize import load_probe, load_data, execute_with_attention_masking, \
    write_html
from explain.bert_components.cmd_nli import ModelConfig
from visualize.html_visual import HtmlVisualizer


def attention_modifier_by_drop_incoming(drop_idx, drop_at):
    def get_modified_attention_mask(attention_mask, layer_no):
        if layer_no != drop_at:
            return attention_mask
        else:
            print("get_modified_attention_mask layer={}".format(layer_no))
            attention_mask_np = np.array(attention_mask)
            # for j in range(len(attention_mask_np)):
            #     attention_mask_np[0, j, drop_idx] = 0
            seq_length = attention_mask_np.shape[1]
            print(seq_length)
            for j in range(1, seq_length):
                attention_mask_np[0, drop_idx, j] = 0

            print('attention_np')
            for i, j in [(0, drop_idx), (drop_idx, 0), (3, drop_idx), (drop_idx, 3)]:
                print(i, j, attention_mask_np[0, i, j])
            modified_attention_mask = tf.constant(attention_mask_np)
            return modified_attention_mask

    return get_modified_attention_mask


def main():
    model_config = ModelConfig()
    model, bert_cls_probe = load_probe(model_config)
    batches = load_data(model_config)
    x0, x1, x2, y = batches[0]
    X = (x0, x1, x2)
    drop_condition_list = [(7, 22), (8, 22), (9, 22), (10, 22), (11, 22), (12, 22)]
    tokenizer = get_tokenizer()

    save_name = "cls_probe_drop_incoming.html"
    html = HtmlVisualizer(save_name)
    for cut_layer, drop_idx in drop_condition_list:
        get_modified_attention_mask = attention_modifier_by_drop_incoming(drop_idx, cut_layer)

        input_ids = x0
        dropped_token = tokenizer.convert_ids_to_tokens(input_ids[:, drop_idx])
        print("Attention dropped tokens:", dropped_token[0])
        messages = ["Drop {}-th token ({}) at layer_{}".format(
            drop_idx, dropped_token[0], cut_layer)]
        logits, probes = execute_with_attention_masking(bert_cls_probe, X, y, get_modified_attention_mask)
        def highlight_term(layer_no_plus_one, seq_idx):
            layer_no = layer_no_plus_one - 1
            # This 'layer_no' includes embedding layer
            if cut_layer == layer_no and drop_idx == seq_idx:
                return " (X)"
            else:
                return ""

        write_html(html, x0, logits, probes, y, messages, highlight_term)


if __name__ == "__main__":
    main()


