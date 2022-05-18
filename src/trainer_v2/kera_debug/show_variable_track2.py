import tensorflow as tf
from official.nlp.keras_nlp import layers

from trainer_v2.train_util.get_tpu_strategy import get_strategy


@tf.keras.utils.register_keras_serializable(package='keras_nlp')
# class BertEncoderModule(tf.Module):
class EmbeddingWrap2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EmbeddingWrap2, self).__init__(kwargs)
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


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        embedding = EmbeddingWrap2()
        input_x = tf.keras.layers.Input(shape=(10,), dtype=tf.int32, name="x")
        print("DEBUG1")
        embedding_out = embedding(input_x)
        self.inner_model = embedding



def main():
    strategy = get_strategy(False, "")
    with strategy.scope():
        model = MyModel()
        for v in model.variables:
            print(v.name)
        print("sub_model_list(tf.keras.layers.Layer):")
        if not model.inner_model.variables:
            print("Sub model has no variables")
        for v in model.inner_model.variables:
            print(v.name)


if __name__ == "__main__":
    main()