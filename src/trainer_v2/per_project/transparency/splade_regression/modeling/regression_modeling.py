from typing import Dict
import tensorflow as tf
from transformers import TFAutoModelForMaskedLM


class ReluSigmoidMaxReduce(tf.keras.layers.Layer):
    def __init__(self):
        super(ReluSigmoidMaxReduce, self).__init__()

    def call(self, out, mask, **kwargs):
        mask_f = tf.expand_dims(tf.cast(mask, tf.float32), axis=2)
        t = tf.nn.relu(out)
        values = tf.reduce_max(tf.math.log(1 + t) * mask_f, axis=1)
        return values


def get_regression_model(model_config: Dict):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    new_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    model = TFAutoModelForMaskedLM.from_pretrained(model_config["model_type"])
    mlm_out = model(new_inputs)
    activation_layer = ReluSigmoidMaxReduce()
    new_out = activation_layer(mlm_out.logits, attention_mask)
    new_model = tf.keras.models.Model(inputs=new_inputs, outputs=[new_out])
    return new_model


def get_regression_model2(mlm_model):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")
    new_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    mlm_out = mlm_model(new_inputs)
    activation_layer = ReluSigmoidMaxReduce()
    new_out = activation_layer(mlm_out['logits'], attention_mask)
    new_model = tf.keras.models.Model(inputs=new_inputs, outputs=[new_out])
    return new_model

