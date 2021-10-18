import numpy as np
import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.bert_components.cls_probe_w_visualize import load_probe, load_data, \
    write_html
from explain.bert_components.cmd_nli import ModelConfig
from explain.bert_components.cmd_nli import logits_to_accuracy
from models.keras_model.bert_keras.bert_common_eager import create_attention_mask_from_input_mask, \
    get_shape_list_no_name, reshape_from_matrix
from models.keras_model.bert_keras.modular_bert import BertClsProbe, reshape_layers_to_3d
from models.transformer.bert_common_v2 import reshape_to_matrix
from visualize.html_visual import HtmlVisualizer


def modify_attention_mask_per_head(attention_scores, attention_mask):
    # `attention_mask` = [B, 1, F, T]
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
    attention_scores += adder
    attention_probs = tf.nn.softmax(attention_scores)

    # print_top_attentions(attention_probs)
    return attention_probs


def print_top_attentions(attention_probs):
    attention_probs_np = np.array(attention_probs)
    source_ranked = np.argsort(attention_probs_np, axis=3)[:, :, :, ::-1]
    src_idx = 22
    for head_idx in range(12):
        s_list = []
        for k in range(10):
            target_idx = source_ranked[0, head_idx, src_idx, k]
            s = "({0}, {1:.2f})".format(target_idx, attention_probs_np[0, head_idx, src_idx, target_idx])
            s_list.append(s)

        print("Head {} : {}".format(head_idx, " ".join(s_list)))


def execute_with_attention_masking(bert_cls: BertClsProbe, X, Y, get_modified_attention_mask):
    input_ids, input_mask, segment_ids = X
    raw_embedding = bert_cls.bert_layer.embedding_layer((input_ids, segment_ids))
    attention_mask = create_attention_mask_from_input_mask(
        input_ids, input_mask)

    embedding = bert_cls.bert_layer.embedding_layer_norm(raw_embedding)
    input_shape = get_shape_list_no_name(embedding)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    prev_output = reshape_to_matrix(embedding)
    shape_info = (batch_size, seq_length, attention_mask)
    all_layer_outputs = []

    for layer_no in range(12):
        layer_input = prev_output
        t_layer = bert_cls.bert_layer.layers[layer_no]
        query = t_layer.query.call(layer_input)
        key = t_layer.key.call(layer_input)
        value = t_layer.value.call(layer_input)

        inputs = query, key, batch_size, seq_length, seq_length
        attention_scores = t_layer.attn_weight.call(inputs)
        modified_attention_mask = get_modified_attention_mask(attention_mask, layer_no)
        attention_probs = modify_attention_mask_per_head(attention_scores, modified_attention_mask)
        context_v = t_layer.context.call((value, attention_probs))
        attention_output = t_layer.attention_output.call(context_v)
        attention_output = t_layer.attention_layer_norm(attention_output + layer_input)

        intermediate_output = t_layer.intermediate.call(attention_output)
        layer_output = t_layer.output_project.call(intermediate_output)
        layer_output = t_layer.output_layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

    final_outputs = reshape_layers_to_3d(all_layer_outputs, input_shape)

    last_layer = reshape_from_matrix(prev_output, input_shape)
    first_token_tensor = tf.squeeze(last_layer[:, 0:1, :], axis=1)
    pooled = bert_cls.pooler(first_token_tensor)
    logits = bert_cls.named_linear(pooled)
    acc = logits_to_accuracy(logits, Y)
    num_probe = 12 + 1
    hidden_v_list = [embedding] + final_outputs
    probe_logit_list = []
    for j in range(num_probe):
        probe_logit = bert_cls.probe_layers[j](hidden_v_list[j])
        probe_logit_list.append(probe_logit)
    return logits, probe_logit_list


def attention_modifier_per_head(drop_idx, drop_at, head_list):
    num_heads = 12

    def get_modified_attention_mask(attention_mask, layer_no):
        if layer_no != drop_at:
            return attention_mask
        else:
            attention_mask_np = np.array(attention_mask)
            attention_mask_np_4d = np.expand_dims(attention_mask_np, 1)
            attention_mask_np_4d = np.tile(attention_mask_np_4d, [1, num_heads, 1, 1])

            seq_length = attention_mask_np.shape[1]
            print(seq_length)
            # for head_idx in range(num_heads):
            for head_idx in head_list:
                for j in range(1, seq_length):
                    attention_mask_np_4d[0, head_idx, drop_idx, j] = 0

            print(attention_mask_np_4d[0, :, drop_idx, :])
            modified_attention_mask = tf.constant(attention_mask_np_4d)
            return modified_attention_mask

    return get_modified_attention_mask


def main():
    model_config = ModelConfig()
    model, bert_cls_probe = load_probe(model_config)
    batches = load_data(model_config)
    x0, x1, x2, y = batches[0]
    X = (x0, x1, x2)
    tokenizer = get_tokenizer()

    save_name = "cls_probe_drop_attention_head.html"
    head_list_list = [list(range(12))]
    # for i in range(12):
    #     head_list_list.append([i])
    head_list_list.extend([[5, 10, 11], [2,3,4], [5,6,8], [1, 5, 11], [3, 4, 10]])
    html = HtmlVisualizer(save_name)
    target_layer = 9
    drop_idx = 22
    for head_list in head_list_list:
        get_modified_attention_mask = attention_modifier_per_head(drop_idx, target_layer, head_list)

        input_ids = x0
        dropped_token = tokenizer.convert_ids_to_tokens(input_ids[:, drop_idx])
        print("Attention dropped tokens:", dropped_token[0])
        messages = ["Drop {}-th token ({})' incoming attention {} at layer_{}".format(
            drop_idx, dropped_token[0], head_list, target_layer)]
        logits, probes = execute_with_attention_masking(bert_cls_probe, X, y, get_modified_attention_mask)
        def highlight_term(layer_no_plus_one, seq_idx):
            layer_no = layer_no_plus_one - 1
            if target_layer == layer_no and drop_idx == seq_idx:
                return " (X)"
            else:
                return ""

        write_html(html, x0, logits, probes, y, messages, highlight_term)


if __name__ == "__main__":
    main()


