from typing import Dict
import tensorflow as tf
from transformers import TFAutoModelForMaskedLM

from trainer_v2.chair_logging import c_log


class ReluSigmoidMaxReduce(tf.keras.layers.Layer):
    def __init__(self):
        super(ReluSigmoidMaxReduce, self).__init__()

    def call(self, out, mask, **kwargs):
        mask_f = tf.expand_dims(tf.cast(mask, tf.float32), axis=2)
        t = tf.nn.relu(out)
        values = tf.reduce_max(tf.math.log(1 + t) * mask_f, axis=1)
        return values


def get_dummy_regression_model(seq_length, _is_training):
    input_ids = tf.keras.layers.Input(shape=(seq_length,), dtype='int32', name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(seq_length,), dtype='int32', name="attention_mask")

    new_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    # w = tf.Variable(np.zeros([30522,], np.float32), dtype=tf.float32, trainable=True)
    #
    # h = tf.cast(tf.reduce_sum(input_ids, axis=1, keepdims=True), tf.float32)
    h = tf.reduce_sum(tf.ones_like(input_ids, tf.float32), axis=1, keepdims=True)
    output = tf.keras.layers.Dense(30522)(h)
    # output = tf.expand_dims(w, axis=0) * tf.expand_dims(h, axis=1)
    new_model = tf.keras.models.Model(inputs=new_inputs, outputs=[output])
    return new_model


def get_transformer_sparse_encoder(model_config: Dict, is_training):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    new_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    model = TFAutoModelForMaskedLM.from_pretrained(model_config["model_type"])
    c_log.info("Initialize model parameter using huggingface: model_type=%s", model_config["model_type"])
    mlm_out = model(new_inputs, training=is_training)
    activation_layer = ReluSigmoidMaxReduce()
    new_out = activation_layer(mlm_out.logits, attention_mask)
    new_model = tf.keras.models.Model(inputs=new_inputs, outputs=[new_out])
    return new_model

