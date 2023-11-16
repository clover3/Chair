from typing import List, Iterable, Callable, Dict, Tuple, Set
from tensorflow import keras
from transformers import AutoTokenizer, TFBertMainLayer
import tensorflow as tf
import numpy as np


from cpath import get_canonical_model_path
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer, get_qd_encoder


class AttentionExtractorHF:
    def __init__(self, model_path, model_config):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.encoder = get_qd_encoder(
            model_config.max_seq_length, is_split_into_words=True)

        c_log.info("Loading model from %s", model_path)
        pair_model = tf.keras.models.load_model(model_path, compile=False)
        self.model = build_attn_extractor(pair_model)

    def predict_list(self, tokens_pair_list: List[Tuple[List, List]]) -> List[np.array]:
        tokens_pair_list_swap = [(b, a) for a, b in tokens_pair_list]
        dataset = self.encoder(tokens_pair_list_swap)
        dataset = dataset.batch(16)
        return self.model.predict(dataset)


def build_attn_extractor(paired_model):
    c_log.info("build_attn_extractor")
    input_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    segment_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
    inputs = (input_ids1, segment_ids1)
    input_1 = {
        'input_ids': input_ids1,
        'token_type_ids': segment_ids1
    }

    old_bert_layer = paired_model.layers[4]
    dense_layer = paired_model.layers[6]
    new_bert_layer = TFBertMainLayer(old_bert_layer._config, name="bert")
    param_values = keras.backend.batch_get_value(old_bert_layer.weights)
    _ = new_bert_layer(get_dummy_input_for_bert_layer())
    keras.backend.batch_set_value(zip(new_bert_layer.weights, param_values))

    bert_output = new_bert_layer(
        input_1, return_dict=True, output_attentions=True)
    logits = dense_layer(bert_output.pooler_output)[:, 0]
    attentions = bert_output.attentions

    attn_probs = tf.stack(attentions, axis=1)  # [B, Layer, H, L, L]
    attn_probs = tf.reduce_mean(tf.reduce_mean(attn_probs, axis=1), axis=1)
    outputs = [attn_probs]
    new_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return new_model


def load_mmp1_attention_extractor():
    model_path = get_canonical_model_path("mmp1")
    return AttentionExtractorHF(model_path, ModelConfig256_1())
