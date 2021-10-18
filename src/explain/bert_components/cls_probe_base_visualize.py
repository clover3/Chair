import os
from typing import List

import numpy as np
import scipy.special
import tensorflow as tf

from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_prediction_summary_str
from cpath import output_path
from data_generator.NLI.nli import get_modified_data_loader
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.bert_components.cmd_nli import logits_to_accuracy, ModelConfig
from models.keras_model.bert_keras.bert_common_eager import create_attention_mask_from_input_mask, \
    get_shape_list_no_name, reshape_from_matrix
from models.keras_model.bert_keras.modular_bert import BertClsProbe
from models.keras_model.bert_keras.modular_unnamed import apply_attention_mask
from models.keras_model.bert_keras.v1_load_util import load_model_cls_probe_from_v1_checkpoint
from models.transformer.bert_common_v2 import reshape_to_matrix
from tlm.token_utils import cells_from_tokens
from trainer.np_modules import get_batches_ex
from visualize.html_visual import HtmlVisualizer, Cell


def write_html(html, input_ids, logits, probe_logits, y):
    num_layers = 12 + 1
    def layer_no_to_name(layer_no):
        if layer_no == 0:
            return "embed"
        else:
            return "layer_{}".format(layer_no-1)
    print(len(probe_logits))
    print(probe_logits[0].shape)
    tokenizer = get_tokenizer()
    num_data = len(input_ids)
    probs_arr = scipy.special.softmax(logits, axis=-1)
    for data_idx in range(num_data)[:100]:
        tokens = tokenizer.convert_ids_to_tokens(input_ids[data_idx])
        first_padding_loc = tokens.index("[PAD]")
        display_len = first_padding_loc + 1
        pred_str = make_prediction_summary_str(probs_arr[data_idx])
        html.write_paragraph("Prediction: {}".format(pred_str))
        html.write_paragraph("gold label={}".format(y[data_idx]))
        text_rows = [Cell("")] + cells_from_tokens(tokens)
        rows = [text_rows]
        mid_pred_rows = []
        for layer_no in range(num_layers):
            layer_logit = probe_logits[layer_no][data_idx]
            probs = scipy.special.softmax(layer_logit, axis=1)
            def prob_to_one_digit(p):
                v = int(p * 10 + 0.05)
                if v > 9:
                    return "A"
                else:
                    s = str(v)
                    assert len(s) == 1
                    return s

            row = [Cell(layer_no_to_name(layer_no))]
            for seq_idx in range(len(probs)):
                case_probs = probs[seq_idx]
                prob_digits: List[str] = list(map(prob_to_one_digit, case_probs))
                cell_str = "".join(prob_digits)
                color_mapping = {
                    0: 2,  # Red = Contradiction
                    1: 1,  # Green = Neutral
                    2: 0   # Blue = Entailment
                }

                color_score = [255 * case_probs[color_mapping[i]] for i in range(3)]
                color = "".join([("%02x" % int(v)) for v in color_score])
                cell = Cell(cell_str, 255, target_color=color)
                row.append(cell)

            mid_pred_rows.append(row)

        rows.extend(mid_pred_rows[::-1])

        rows = [row[:display_len] for row in rows]
        html.write_table(rows)


def load_probe(model_config):
    save_path = os.path.join(output_path, "model", "runs", "nli_probe_cls3", "model-100000")
    model, bert_cls_probe = load_model_cls_probe_from_v1_checkpoint(save_path, model_config)
    return model, bert_cls_probe


def load_data(model_config: ModelConfig):
    vocab_filename = "bert_voca.txt"
    data_loader = get_modified_data_loader(get_tokenizer(), model_config.max_seq_length, vocab_filename)
    dev_insts = data_loader.get_dev_data()
    return get_batches_ex(dev_insts, 128, 4)


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


def main():
    model_config = ModelConfig()
    model, bert_cls_probe = load_probe(model_config)
    # bert_cls_probe, dev_insts = load_cls_probe()
    # batches = get_batches_ex(dev_insts, 32, 4)
    batches = load_data(model_config)
    x0, x1, x2, y = batches[0]
    X = (x0, x1, x2)
    save_name = "cls_probe_base_local.html"
    html = HtmlVisualizer(save_name)
    logits, probes = bert_cls_probe(X)
    write_html(html, x0, logits, probes, y)



if __name__ == "__main__":
    main()


