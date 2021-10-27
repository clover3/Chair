import numpy as np
import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.bert_components.cmd_nli import logits_to_accuracy, ModelConfig
from models.keras_model.bert_keras.bert_common_eager import create_attention_mask_from_input_mask, \
    get_shape_list_no_name, reshape_from_matrix
from models.keras_model.bert_keras.modular_bert import BertClsProbe, reshape_layers_to_3d
from models.keras_model.bert_keras.modular_unnamed import apply_attention_mask
from models.transformer.bert_common_v2 import reshape_to_matrix


def execute_with_attention_logging(bert_cls: BertClsProbe, X, Y, attention_logger):
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
        attention_scores = apply_attention_mask(attention_scores, attention_mask)
        attention_probs = tf.nn.softmax(attention_scores)
        attention_logger(layer_no, attention_probs)
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



def get_attention_logger(input_ids, target_idx):
    tokenizer = get_tokenizer()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print("Target token: {}({})".format(tokens[target_idx], target_idx))

    def get_attention_summary(probs):
        ranked = np.argsort(probs)[::-1]
        s_list = []
        for j in ranked:
            if tokens[j] not in ["[SEP]", "[CLS]", "[PAD]"]:
                s = get_token_desc(j, probs)
                s_list.append(s)
            if len(s_list) > 10:
                break
        return " ".join(s_list)

    def get_attention_summary_all(probs):
        ranked = np.argsort(probs)[::-1]
        s_list = []
        for j in ranked:
            if tokens[j] not in ["[SEP]", "[CLS]", "[PAD]"]:
                s = get_token_desc(j, probs)
                s_list.append(s)
            if probs[j] < 1:
                break
        return " ".join(s_list)

    def get_token_desc(j, probs):
        s = "{0}[{1}]({2:.2f})".format(tokens[j], j, probs[j])
        return s

    attention_probs_np_list = []
    def log_attention(layer_no, attention_probs):
        attention_probs_np = np.array(attention_probs)
        attention_probs_np_list.append(attention_probs_np[0])
        print("Layer {}".format(layer_no))
        for head_no in range(12):
            incoming = attention_probs_np[0, head_no, target_idx, :]
            outgoing = attention_probs_np[0, head_no, :, target_idx]
            print("Head {}".format(head_no))
            print("incoming", get_attention_summary(incoming))
            print("outgoing", get_attention_summary(outgoing))

        if layer_no == 11:
            print("Summary")
            all_layer_attention = np.stack(attention_probs_np_list, axis=0)
            attention_sum = np.sum(np.sum(all_layer_attention, axis=1), axis=0)
            print("incoming", get_attention_summary_all(attention_sum[target_idx, :]))
            print("outgoing", get_attention_summary_all(attention_sum[:, target_idx]))

    return log_attention


def main():
    model_config = ModelConfig()
    from explain.bert_components.cls_probe_w_visualize import load_probe
    model, bert_cls_probe = load_probe(model_config)
    from explain.bert_components.cls_probe_w_visualize import load_data
    batches = load_data(model_config)
    x0, x1, x2, y = batches[0]
    hooking_fn = get_attention_logger(x0, 23)
    X = (x0, x1, x2)
    execute_with_attention_logging(bert_cls_probe, X, y, hooking_fn)


if __name__ == "__main__":
    main()


