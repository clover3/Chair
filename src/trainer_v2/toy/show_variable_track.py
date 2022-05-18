import tensorflow as tf
from official.nlp.keras_nlp import layers

from trainer_v2.train_util.get_tpu_strategy import get_strategy


#
#
# class OnDeviceEmbedding2(tf.keras.layers.Layer):
#     def __init__(self,
#                  vocab_size,
#                  embedding_width,
#                  initializer="glorot_uniform",
#                  use_one_hot=False,
#                  scale_factor=None,
#                  **kwargs):
#
#         super(OnDeviceEmbedding2, self).__init__(**kwargs)
#         self._vocab_size = vocab_size
#         self._embedding_width = embedding_width
#         self._initializer = initializer
#         self._use_one_hot = use_one_hot
#         self._scale_factor = scale_factor
#         self.hidden = tf.keras.layers.Dense(10, activation='relu')
#         self.embeddings = self.add_weight(
#             "embeddings2",
#             shape=[self._vocab_size, self._embedding_width],
#             initializer=self._initializer,
#             dtype=tf.float32)
#         print("init is called?")
#
#     # def build(self, input_shape):
#     #     super(OnDeviceEmbedding2, self).build(input_shape)
#
#     def call(self, inputs):
#         print("call is called?")
#         flat_inputs = tf.reshape(inputs, [-1])
#         if self._use_one_hot:
#             dtype = self._compute_dtype
#             if not tf.dtypes.as_dtype(dtype).is_floating:
#                 # TensorFlow 1 compatibility. In TF1, self._compute_dtype is int32
#                 # instead of a floating-point dtype, as the dtype is inferred from the
#                 # dtype of the inputs
#                 dtype = tf.float32
#             one_hot_data = tf.one_hot(
#                 flat_inputs, depth=self._vocab_size, dtype=dtype)
#             embeddings = tf.matmul(one_hot_data, self.embeddings)
#         else:
#             embeddings = tf.gather(self.embeddings, flat_inputs)
#         embeddings = tf.reshape(
#             embeddings,
#             # Work around b/142213824: prefer concat to shape over a Python list.
#             tf.concat([tf.shape(inputs), [self._embedding_width]], axis=0))
#         embeddings.set_shape(inputs.shape.as_list() + [self._embedding_width])
#         if self._scale_factor:
#             embeddings *= self._scale_factor
#         return embeddings


class EmbeddingWrap(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EmbeddingWrap, self).__init__(kwargs)
        self.embedding = layers.OnDeviceEmbedding(10, 728)

    def call(self, inputs):
        return self.embedding(inputs)


class MLPWithEmbedding(tf.keras.Model):
    def __init__(self, **kwargs):
        super(MLPWithEmbedding, self).__init__()
        embedding = EmbeddingWrap(
            vocab_size=200,
            embedding_width=10,
            name='word_embeddings')

        input_x = tf.keras.layers.Input(shape=(10,), dtype=tf.int32, name="x")
        embedding_out = embedding(input_x)
        self.inner_model = embedding


def main():
    strategy = get_strategy(False, "")
    with strategy.scope():
        outer_model = MLPWithEmbedding()
        for v in outer_model.variables:
            print(v.name)
        print("inner_module EmbeddingWrap(tf.keras.layers.Layer):")
        for v in outer_model.inner_model.variables:
            print(v.name)


if __name__ == "__main__":
    main()