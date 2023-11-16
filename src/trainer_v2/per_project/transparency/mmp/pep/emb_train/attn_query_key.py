import numpy as np
import tensorflow as tf
from collections import namedtuple

from bert.embeddings import BertEmbeddingsLayer
from bert.layer import Layer
from tensorflow import keras
from tensorflow.python.keras import backend as K


from cpath import get_bert_config_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, define_bert_input
from trainer_v2.per_project.cip.tfrecord_gen import pad_to_length

QueryKeyValue = namedtuple("QueryKeyValue", ['query', 'key', "value"])


class OneLayerAttnQueryKey(Layer):
    class Params(Layer.Params):
        num_heads         = None
        size_per_head     = None
        initializer_range = 0.02
        query_activation  = None
        key_activation    = None
        value_activation  = None
        num_layers = None

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.query_activation = self.params.query_activation
        self.key_activation   = self.params.key_activation
        self.value_activation = self.params.value_activation

        self.supports_masking = True
        self.qkv_list = []

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        dense_units = self.params.num_heads * self.params.size_per_head  # N*H
        query_layer = keras.layers.Dense(units=dense_units, activation=self.query_activation,
                                         kernel_initializer=self.create_initializer(),
                                         name="query")
        key_layer = keras.layers.Dense(units=dense_units, activation=self.key_activation,
                                       kernel_initializer=self.create_initializer(),
                                       name="key")
        value_layer = keras.layers.Dense(units=dense_units, activation=self.value_activation,
                                         kernel_initializer=self.create_initializer(),
                                         name="value")
        self.qkv = QueryKeyValue(query_layer, key_layer, value_layer)

    def call(self, inputs):
        input_shape = tf.shape(input=inputs)
        batch_size, seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]

        def transpose_for_scores(input_tensor, seq_len):
            output_shape = [batch_size, seq_len,
                            self.params.num_heads, self.params.size_per_head]
            output_tensor = K.reshape(input_tensor, output_shape)
            return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,F,H]

        def reshape_scores(input_tensor):
            output_shape = [batch_size, self.params.num_heads, self.params.size_per_head]
            output_tensor = K.reshape(input_tensor, output_shape)
            return output_tensor # [B, N, H]

        q_v = transpose_for_scores(self.qkv.query(inputs), seq_len)
        k_v = transpose_for_scores(self.qkv.key(inputs), seq_len)
        v_v = transpose_for_scores(self.qkv.value(inputs), seq_len)
        vectors = QueryKeyValue(q_v, k_v, v_v)
        return vectors


class AttnQueryKey(Layer):
    class Params(Layer.Params):
        num_heads         = None
        size_per_head     = None
        initializer_range = 0.02
        query_activation  = None
        key_activation    = None
        value_activation  = None
        num_layers = None

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.query_activation = self.params.query_activation
        self.key_activation   = self.params.key_activation
        self.value_activation = self.params.value_activation

        self.supports_masking = True
        self.qkv_list = []

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        qkv_list = []
        for i in range(self.params.num_layers):
            one_qkv = OneLayerAttnQueryKey.from_params(self.params, name="layer_{}".format(i))
            qkv_list.append(one_qkv)
        self.qkv_list = qkv_list

    def call(self, inputs):
        input_shape = tf.shape(input=inputs)
        out_vectors = []
        for i, qkv in enumerate(self.qkv_list):
            vectors = qkv(inputs)
            out_vectors.append(vectors)
        return out_vectors


def load_keras_model_weights(
        model, keras_model_path, name_mapping, expected_n_restored=None):
    # Load the entire model
    loaded_model = tf.keras.models.load_model(keras_model_path, compile=False)

    # Extract weights from the loaded model
    ckpt_weights = {w.name: w for w in loaded_model.weights}
    weight_value_tuples = []
    loaded_weights = set()

    # Go through the current model's weights and replace them
    for weight in model.weights:
        mapped_name = name_mapping(weight.name)
        if mapped_name in ckpt_weights:
            ckpt_var = keras.backend.get_value(ckpt_weights[mapped_name])
            loaded_weights.add(mapped_name)
            weight_value_tuples.append((weight, ckpt_var))
        else:
            print(f"Warning: No weight found for {weight.name} i.e.:[{mapped_name}]")


    keras.backend.batch_set_value(weight_value_tuples)
    loaded_weights_names = set(ckpt_weights.keys())
    unused_weights = sorted(loaded_weights_names.difference(loaded_weights))

    if expected_n_restored is None:
        if len(weight_value_tuples) != expected_n_restored:
            raise ValueError("{} is expected but only restored {}".format(expected_n_restored, len(weight_value_tuples)))
        print("Unused weights from checkpoint:",
              "\n\t" + "\n\t".join(unused_weights))


def get_bert_qkv_encoder(ckpt_path, seq_len):
    bert_params = load_bert_config(get_bert_config_path())
    size_per_head = bert_params.hidden_size // bert_params.num_heads
    qkv_layer = AttnQueryKey.from_params(
        bert_params,
        size_per_head=size_per_head,
    )

    embedding_layer = BertEmbeddingsLayer.from_params(bert_params, name="embeddings")

    l_input_ids, l_token_type_ids = define_bert_input(seq_len, "")
    inputs = [l_input_ids, l_token_type_ids]

    emb_vector = embedding_layer(inputs)
    output = qkv_layer(emb_vector)

    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    # 1. Load params
    expected_n_restored = 12 * 3 * 2 + 3 + 2

    def name_mapping(name):
        out_name = name
        if name.startswith("attn_query_key"):
            _, layer_no, qkv, kernel_bias = name.split("/")
            out_name = f"encoder/bert/encoder/{layer_no}/attention/self/{qkv}/{kernel_bias}"
        else:
            n1, n2, n3 = name.split("/")
            out_name = f"encoder/bert/{n1}/{n2}/{n3}"
        return out_name

    load_keras_model_weights(model, ckpt_path, name_mapping, expected_n_restored)
    return model


def get_predictor(ckpt_path):
    c_log.info("Loading model from %s", ckpt_path)
    tokenizer = get_tokenizer()
    segment_len = 256
    model = get_bert_qkv_encoder(ckpt_path, segment_len)

    def predict_rep(tokens: list[str], is_query: bool) -> list[QueryKeyValue]:
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_type = 0 if is_query else 1
        segment_ids = [token_type] * len(tokens)
        input_ids = pad_to_length(input_ids, segment_len)
        segment_ids = pad_to_length(segment_ids, segment_len)

        def generator():
            yield (input_ids, segment_ids)

        int_list = tf.TensorSpec(shape=(segment_len,), dtype=tf.int32)
        output_signature = (int_list, int_list)
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.batch(1)

        def reform(x1, x2):
            return (x1, x2), tf.zeros([0], tf.int32)

        dataset = dataset.map(reform)
        output = model.predict(dataset)
        rep_list: list[QueryKeyValue] = []
        for token_idx in range(len(tokens)):
            t = []
            for tuple_idx in range(3):
                rep_per_tuple = np.array([output[layer][tuple_idx][0, :, token_idx, :] for layer in range(12)])
                t.append(rep_per_tuple)
            rep_list.append(QueryKeyValue(*t))
            # [12, num_head, dim_per_head]
        return rep_list

    return predict_rep