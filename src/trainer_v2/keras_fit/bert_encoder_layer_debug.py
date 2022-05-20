import tensorflow as tf
from official.nlp.keras_nlp import layers


@tf.keras.utils.register_keras_serializable(package='keras_nlp')
# class BertEncoderModule(tf.Module):
class TestBertEncoderModule(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TestBertEncoderModule, self).__init__(kwargs)
        self._self_setattr_tracking = False
        self.embedding_layer: tf.keras.layers.Layer = self._build_embedding_layer()

    def call(self, word_ids):
        word_embeddings = self.embedding_layer(word_ids)
        return word_embeddings[:, 0]

    def _build_embedding_layer(self):
        return layers.OnDeviceEmbedding(
            vocab_size=10,
            embedding_width=728,
            name='word_embeddings')

