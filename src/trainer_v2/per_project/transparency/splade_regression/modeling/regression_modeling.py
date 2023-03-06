import tensorflow as tf
from transformers import TFAutoModelForMaskedLM

from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay


class ReluSigmoidMaxReduce(tf.keras.layers.Layer):
    def __init__(self):
        super(ReluSigmoidMaxReduce, self).__init__()

    def call(self, out, mask, **kwargs):
        mask_f = tf.expand_dims(tf.cast(mask, tf.float32), axis=2)
        t = tf.nn.relu(out)
        values = tf.reduce_max(tf.math.log(1 + t) * mask_f, axis=1)
        return values


def get_regression_model(run_config):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
    new_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    model = TFAutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
    mlm_out = model(new_inputs)
    activation_layer = ReluSigmoidMaxReduce()
    new_out = activation_layer(mlm_out.logits, attention_mask)
    new_model = tf.keras.models.Model(inputs=new_inputs, outputs=[new_out])
    new_model.summary()
    optimizer = AdamWeightDecay(learning_rate=run_config.train_config.learning_rate)
    new_model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    return new_model