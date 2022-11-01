from cpath import get_bert_config_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from models.transformer.bert_common_v2 import get_shape_list2
from tab_print import print_table
from trainer_v2.custom_loop.attention_helper.attention_extractor import InferenceHelper, get_layer_by_class_name, \
    AttentionScoresDetailed
from trainer_v2.custom_loop.attention_helper.model_shortcut import load_nli14_attention_extractor
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.modeling_common.network_utils import split_stack_input
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerSingle
from trainer_v2.custom_loop.neural_network_def.two_seg_concat_sap import TwoSegConcat2SAP
from trainer_v2.custom_loop.runner.two_seg_concat2 import ModelConfig as ModelConfig600
from data_generator2.segmented_enc.seg_encoder_common import TwoSegConcatEncoder
import tensorflow as tf
import numpy as np

from trainer_v2.custom_loop.train_loop import load_model_by_dir_or_abs


def get_input_splitter(total_seq_length, window_length):
    num_window = int(total_seq_length / window_length)
    assert total_seq_length % window_length == 0
    def input_splitter(input_list):
        batch_size, _ = get_shape_list2(input_list[0])

        def r3to2(arr):
            return tf.reshape(arr, [batch_size * num_window, window_length])

        input_list_stacked = split_stack_input(input_list, total_seq_length, window_length)
        input_list_flatten = list(map(r3to2, input_list_stacked))  # [batch_size * num_window, window_length]
        return input_list_flatten
    return input_splitter


def with_nli():
    p = "The documents in government storage can be searched."
    h = "The documents are not searchable"
    ae = load_nli14_attention_extractor()
    tokenizer = get_tokenizer()
    p_tokens = tokenizer.tokenize(p)
    h_tokens = tokenizer.tokenize(h)

    attn_score_list = ae.predict_list([(p_tokens, h_tokens)])
    assert len(attn_score_list) == 1
    attn_score: AttentionScoresDetailed = attn_score_list[0]

    n_tokens = len(p_tokens) + len(h_tokens) + 3
    attn_acc = attn_score.get_layer_head_merged()
    seq = ["CLS"] + p_tokens + ["SEP"] + h_tokens + ["SEP"]

    head = [""] + seq
    table = [head]
    for j in range(n_tokens):
        row = [seq[j]]
        for k in range(n_tokens):
            row.append("{0:.2f}".format(attn_acc[j, k]))
        table.append(row)
    print_table(table)


def with_nlits():
    p = "The documents in government storage can be searched."
    h = "The documents are not searchable"
    bert_params = load_bert_config(get_bert_config_path())
    model_path = "C:\\work\\code\\Chair\\output\\model\\runs\\nli_ts_run40_0\\model_25000"
    model_config = ModelConfig600()

    tokenizer = get_tokenizer()
    encoder = TwoSegConcatEncoder(tokenizer, model_config.max_seq_length)
    window_length = int(model_config.max_seq_length / 2)
    src_model = load_model_by_dir_or_abs(model_path)
    tsc = TwoSegConcat2SAP(FuzzyLogicLayerSingle)
    tsc.build_model(bert_params, model_config)
    tsc_inner = tsc.get_keras_model()
    tsc_inner.set_weights(src_model.get_weights())
    bert_sap = get_layer_by_class_name(tsc_inner, "BertModelLayerSAP")

    new_outputs = tsc_inner.outputs + list(bert_sap.output)
    new_model = tf.keras.models.Model(inputs=tsc_inner.inputs, outputs=new_outputs, name="SAP")

    p_tokens = tokenizer.tokenize(p)
    h_tokens = tokenizer.tokenize(h)
    cut_at = 3
    h_tokens1 = h_tokens[:cut_at]
    h_tokens2 = h_tokens[cut_at:]
    helper = InferenceHelper(new_model)
    triplet = encoder.two_seg_concat_core(p_tokens, h_tokens1, h_tokens2)
    input_ids, segment_ids, _ = triplet
    x_list = [(input_ids, segment_ids)]
    outputs = helper.predict(x_list)
    g_decision, encoding_output, attention_probs = outputs
    print("g_decision", g_decision)
    print("attention_probs[0]", attention_probs[0].shape)
    print("h_token1", h_tokens1)
    print("h_token2", h_tokens2)

    attn_avg_over_layers = np.mean(np.stack(attention_probs, 0), axis=0)

    h_tokens_seg = [h_tokens1, h_tokens2]
    for seg_no in range(2):
        print("h_tokens{}".format(seg_no+1))
        n_tokens = len(p_tokens) + len(h_tokens_seg[seg_no]) + 3
        attn = attn_avg_over_layers[seg_no]
        print("attn", attn.shape)
        attn_acc = np.mean(attn, axis=0)
        seq = ["CLS"] + p_tokens + ["SEP"] + h_tokens_seg[seg_no] + ["SEP"]

        head = [""] + seq
        table = [head]
        for j in range(n_tokens):
            row = [seq[j]]
            for k in range(n_tokens):
                row.append("{0:.2f}".format(attn_acc[j, k]))
            table.append(row)
        print_table(table)


if __name__ == "__main__":
    with_nli()
