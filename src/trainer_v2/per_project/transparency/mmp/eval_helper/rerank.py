from typing import List
from typing import List, Iterable, Callable, Dict, Tuple, Set

from transformers import TFBertMainLayer
from tensorflow import keras

from trainer_v2.chair_logging import c_log
import tensorflow as tf

from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_qd_encoder, get_dummy_input_for_bert_layer


def build_inference_model(paired_model):
    input_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids1")
    segment_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids1")
    inputs = [input_ids1, segment_ids1]
    input_1 = {
        'input_ids': input_ids1,
        'token_type_ids': segment_ids1
    }

    bert_layer = paired_model.layers[4]
    dense_layer = paired_model.layers[6]

    bert_output = bert_layer(input_1)
    logits = dense_layer(bert_output['pooler_output'])[:, 0]
    new_model = tf.keras.models.Model(inputs=inputs, outputs=logits)
    return new_model


def build_inference_model2(paired_model):
    c_log.info("build_inference_model2")
    input_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    segment_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
    inputs = [input_ids1, segment_ids1]
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

    bert_output = new_bert_layer(input_1)

    logits = dense_layer(bert_output['pooler_output'])[:, 0]
    new_model = tf.keras.models.Model(inputs=inputs, outputs=logits)
    return new_model


def get_scorer(model_path, batch_size=16):
    max_seq_length = 256
    c_log.info("Loading model from %s", model_path)
    paired_model = tf.keras.models.load_model(model_path, compile=False)
    inference_model = build_inference_model2(paired_model)

    qd_encoder = get_qd_encoder(max_seq_length)

    def score_fn(qd_list: List):
        dataset = qd_encoder(qd_list)
        dataset = dataset.batch(batch_size)
        output = inference_model.predict(dataset)
        return output

    c_log.info("Defining network")
    return score_fn

