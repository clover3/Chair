import keras
import tensorflow as tf


class WeightedSum(keras.layers.Layer):
    def __init__(self):
        super(WeightedSum, self).__init__()

    def call(self, args):
        x = args[0]
        m = args[1]
        s = tf.reduce_sum(x, axis=1)
        d = tf.reduce_sum(tf.cast(tf.equal(m, 0), tf.float32), axis=-1)
        s = s / tf.expand_dims(d, 1)
        return s


def make_embedding_layer(params, name: str = 'embedding',) -> keras.layers.Layer:
    return keras.layers.Embedding(
        params['embedding_input_dim'],
        params['embedding_output_dim'],
        trainable=params['embedding_trainable'],
        name=name,
    )


def build_model(word_index, embedding_matrix, embedding_dim, max_seq_length):
    embedding = make_embedding_layer()
    model = keras.Model(inputs=[query, doc], outputs=[out])
