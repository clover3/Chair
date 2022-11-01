from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple
from cpath import pjoin, data_path, get_bert_config_path
from data_generator.tokenizer_wo_tf import get_tokenizer, EncoderUnitPlain
from data_generator2.segmented_enc.seg_encoder_common import BasicConcatEncoder
from misc_lib import ceil_divide
import numpy as np
import tensorflow as tf

from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.neural_network_def.sap_bert import BertSAP
from trainer_v2.custom_loop.train_loop import load_model_by_dir_or_abs


# This part does not know if input has two segments or not.
# Responsible for: Loading model, running prediction
KerasLayer = tf.keras.layers.Layer


def rebuild_model(source, target):
    target.set_weights(source.get_weights())


class InferenceHelper:
    def __init__(self, model, strategy=None, batch_size = 16):
        def get_spec(input_i: tf.keras.layers.Input):
            return tf.TensorSpec(input_i.type_spec.shape[1:], dtype=tf.int32)
        self.out_sig = [get_spec(input) for input in model.inputs]
        self.n_input = len(model.inputs)
        self.model = model
        self.strategy = strategy
        self.batch_size = batch_size

    def predict(self, input_list):
        def generator():
            yield from input_list

        def reform(*row):
            # return (*row),
            if self.n_input == 4:
                x = row[0], row[1], row[2], row[3]
            elif self.n_input == 2:
                x = row[0], row[1],
            else:
                raise ValueError
            return x,

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=tuple(self.out_sig))
        dataset = dataset.map(reform)

        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        maybe_step = ceil_divide(len(input_list), self.batch_size)
        if self.strategy is not None:
            dataset = distribute_dataset(self.strategy, dataset)
        model = self.model
        l_decision = model.predict(dataset, steps=maybe_step)
        return l_decision


class AttentionScoresDetailed:
    def __init__(self, attention_probs):
        # attention_probs[layer_no][head_no][from_idx][to_idx]
        assert len(attention_probs.shape) == 4
        self.attention_probs = attention_probs

    def get_layer_head_merged(self):
        layer_merged = np.mean(self.attention_probs, axis=0)
        return np.mean(layer_merged, axis=0)

    def get_one_head(self, layer_no, head_no):
        return self.attention_probs[layer_no, head_no]


class AttentionExtractorInner:
    def __init__(self, model_path, model_config, num_layer=12):
        bert = BertSAP()
        bert_params = load_bert_config(get_bert_config_path())
        bert.build_model(bert_params, model_config)
        tsc_inner = bert.get_keras_model()
        src_model = load_model_by_dir_or_abs(model_path)
        tsc_inner.set_weights(src_model.get_weights())
        bert_sap = get_layer_by_class_name(tsc_inner, "BertModelLayerSAP")
        seq_output, attn_probs = bert_sap.output
        new_outputs = tsc_inner.outputs + [attn_probs]
        new_model = tf.keras.models.Model(inputs=tsc_inner.inputs, outputs=new_outputs, name="SAP")
        self.model = new_model
        self.inference_helper = InferenceHelper(new_model)
        self.num_layer = num_layer

    def predict_list(self, triplet_list: List[Tuple[List, List, List]]) -> List[AttentionScoresDetailed]:
        x_list = []
        for input_ids, segment_ids, _ in triplet_list:
            x_list.append((input_ids, segment_ids))
        outputs = self.inference_helper.predict(x_list)
        g_decision, attention_probs = outputs
        n_item = len(g_decision)
        output = []
        for i in range(n_item):
            probs_per_layer = []
            for layer_no in range(self.num_layer):
                probs = attention_probs[layer_no][i]  # [Batch, num_heads, seq_len, seq_len
                probs_per_layer.append(probs)

            probs_per_layer = np.stack(probs_per_layer, axis=0)
            attn_score = AttentionScoresDetailed(probs_per_layer)
            output.append(attn_score)
        return output


class AttentionExtractorSummed:
    def __init__(self, model_path, model_config, num_layer=12):
        bert = BertSAP()
        bert_params = load_bert_config(get_bert_config_path())
        bert.build_model(bert_params, model_config)
        tsc_inner = bert.get_keras_model()
        src_model = load_model_by_dir_or_abs(model_path)
        tsc_inner.set_weights(src_model.get_weights())
        bert_sap = get_layer_by_class_name(tsc_inner, "BertModelLayerSAP")
        seq_output, attn_probs = bert_sap.output

        attn_probs = tf.stack(attn_probs, axis=1)  # [B, Layer, H, L, L]
        attn_probs = tf.reduce_mean(tf.reduce_mean(attn_probs, axis=1), axis=1)
        new_outputs = tsc_inner.outputs + [attn_probs]
        new_model = tf.keras.models.Model(inputs=tsc_inner.inputs, outputs=new_outputs, name="SAP")
        self.model = new_model
        self.inference_helper = InferenceHelper(new_model)

    def predict_list(self, triplet_list: List[Tuple[List, List, List]]) -> List[np.array]:
        x_list = []
        for input_ids, segment_ids, _ in triplet_list:
            x_list.append((input_ids, segment_ids))
        outputs = self.inference_helper.predict(x_list)
        g_decision, attention_probs = outputs
        n_item = len(g_decision)
        output = []
        for i in range(n_item):
            output.append(attention_probs[i])
        return output


class AttentionExtractor:
    def __init__(self, model_path, model_config, num_layer=12):
        self.encoder = BasicConcatEncoder(get_tokenizer(), model_config.max_seq_length)
        self.inner = AttentionExtractorSummed(model_path, model_config, num_layer)

    def predict_list(self, tokens_pair_list: List[Tuple[List, List]]) -> List[np.array]:
        triplet_list = []
        for tokens1, tokens2 in tokens_pair_list:
            triplet = self.encoder.encode(tokens1, tokens2)
            triplet_list.append(triplet)
        return self.inner.predict_list(triplet_list)


def get_bert_layer(model) -> tf.keras.layers.Layer:
    for idx, layer in enumerate(model.layers):
        if type(layer).__name__ == "BertModelLayer":
            print("Using {} as BertModelLayer".format(layer))
            return layer
    raise ValueError


def get_layer_by_class_name(model, name) -> tf.keras.layers.Layer:
    for idx, layer in enumerate(model.layers):
        if type(layer).__name__ == name:
            print("Using {} as BertModelLayer".format(layer))
            return layer
    raise ValueError
