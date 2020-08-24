import keras
import tensorflow as tf
from matchzoo.engine.base_model import BaseModel
from matchzoo.models import KNRM

from models.keras_model.AverageEmbedding import WeightedSum


def layer_norm(input_tensor):
    return tf.keras.layers.LayerNormalization(epsilon=1e-3, axis=-1)(input_tensor)


class AvgEmbedding(BaseModel):

    @classmethod
    def get_default_params(cls):
        """Get default parameters."""
        params = super().get_default_params(with_embedding=True)
        return params

    def weighted_sum_layer(self):
        def fn(x, m):
            s = tf.reduce_sum(x, axis=1)
            d = tf.sum(tf.cast(tf.equal(m, 0), tf.float32), axis=-1)
            s = s / d
            return s
        return keras.layers.Lambda(fn)

    def build(self):
        """Build model."""
        query, doc = self._make_inputs()
        embedding = self._make_embedding_layer()
        q_embed = embedding(query)
        d_embed = embedding(doc)
        layer = WeightedSum()
        q_embed = layer([q_embed, query])
        d_embed = layer([d_embed, doc])
        mm = keras.layers.Dot(axes=1, normalize=True)([q_embed, d_embed])
        mm = keras.layers.Lambda(lambda x: tf.reshape(x, [-1, 1]))(mm)
        out = keras.layers.Dense(2)(mm)
        self._backend = keras.Model(inputs=[query, doc], outputs=[out])


class KNRMEx(KNRM):
    def build(self):
        query, doc = self._make_inputs()

        embedding = self._make_embedding_layer()
        q_embed = embedding(query)
        d_embed = embedding(doc)

        def residual(x):
            xp = keras.layers.Dense(self._params['embedding_output_dim'])(x)
            x = xp + x
            return x

        # q_embed = layer_norm(q_embed)
        # d_embed = layer_norm(d_embed)

        q_embed = residual(q_embed)
        d_embed = residual(d_embed)

        mm = keras.layers.Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])

        KM = []
        for i in range(self._params['kernel_num']):
            mu = 1. / (self._params['kernel_num'] - 1) + (2. * i) / (
                self._params['kernel_num'] - 1) - 1.0
            sigma = self._params['sigma']
            if mu > 1.0:
                sigma = self._params['exact_sigma']
                mu = 1.0
            mm_exp = self._kernel_layer(mu, sigma)(mm)
            mm_doc_sum = keras.layers.Lambda(
                lambda x: tf.reduce_sum(x, 2))(mm_exp)
            mm_log = keras.layers.Activation(tf.math.log1p)(mm_doc_sum)
            mm_sum = keras.layers.Lambda(
                lambda x: tf.reduce_sum(x, 1))(mm_log)
            KM.append(mm_sum)

        phi = keras.layers.Lambda(lambda x: tf.stack(x, 1))(KM)
        hidden = keras.layers.Dense(10, activation='relu')(phi)
        out = self._make_output_layer()(hidden)
        self._backend = keras.Model(inputs=[query, doc], outputs=[out])
