import numpy as np
import tensorflow as tf

from explain.bert_components.cmd_nli import logits_to_accuracy
from models.keras_model.bert_keras.bert_common_eager import create_attention_mask_from_input_mask, \
    get_shape_list_no_name, reshape_from_matrix
from models.keras_model.bert_keras.modular_bert import BertClsProbe
from models.keras_model.bert_keras.modular_unnamed import apply_attention_mask
from models.transformer.bert_common_v2 import reshape_to_matrix


def execute_with_attention_masking(bert_cls: BertClsProbe, X, Y, cut_layer, drop_idx):
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

    for i in range(cut_layer):
        prev_output = bert_cls.bert_layer.layers[i]((prev_output, shape_info))
        all_layer_outputs.append(prev_output)

    attention_mask_np = np.array(attention_mask)

    # for j in range(len(attention_mask_np)):
    #     attention_mask_np[0, j, drop_idx] = 0
    attention_mask_np[0, 0, drop_idx] = 0
    print(attention_mask_np)
    modified_attention_mask = tf.constant(attention_mask_np)

    for i in range(cut_layer, 12):
        layer_input = prev_output
        t_layer = bert_cls.bert_layer.layers[i]
        query = t_layer.query.call(layer_input)
        key = t_layer.key.call(layer_input)
        value = t_layer.value.call(layer_input)

        inputs = query, key, batch_size, seq_length, seq_length
        attention_scores = t_layer.attn_weight.call(inputs)
        attention_scores = apply_attention_mask(attention_scores, modified_attention_mask)
        attention_probs = tf.nn.softmax(attention_scores)
        context_v = t_layer.context.call((value, attention_probs))
        attention_output = t_layer.attention_output.call(context_v)
        attention_output = t_layer.attention_layer_norm(attention_output + layer_input)

        intermediate_output = t_layer.intermediate.call(attention_output)
        layer_output = t_layer.output_project.call(intermediate_output)
        layer_output = t_layer.output_layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)
    final_outputs = []
    for layer_output in all_layer_outputs:
        final_output = reshape_from_matrix(layer_output, input_shape)
        final_outputs.append(final_output)

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