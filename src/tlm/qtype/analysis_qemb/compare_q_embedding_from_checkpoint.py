import os
from collections import OrderedDict

import tensorflow as tf
from tensorflow_core.python.training.checkpoint_utils import load_checkpoint

from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from models.keras_model.bert_keras.bert_common_eager import get_shape_list_no_name


def apply_layer_norm(x, gamma, beta, epsilon):
    if tf.__version__.startswith("2."):
        mean, var = tf.nn.moments(x, axes=-1, keepdims=True)
    else:
        mean, var = tf.nn.moments(x, axes=-1, keep_dims=True)
    inv = gamma * tf.math.rsqrt(var + epsilon)
    res = x * tf.cast(inv, x.dtype) + tf.cast(beta - mean * inv, x.dtype)
    return res


def main():
    checkpoint_path = os.path.join(output_path, "model", "runs", "mmd_qtype_O", "model.ckpt-20000")
    d = OrderedDict()
    name_word_embedding = "SCOPE1/bert/embeddings/word_embeddings"
    name_qtype_embedding = "SCOPE2/qtype_modeling/qtype_embedding"
    reader = load_checkpoint(checkpoint_path)
    #
    # for x in tf.train.list_variables(checkpoint_path):
    #     (name, var) = (x[0], x[1])
    #     print(name)
    # return

    word_embedding = reader.get_tensor(name_word_embedding)
    qtype_embedding = reader.get_tensor(name_qtype_embedding)
    query_bias = reader.get_tensor("SCOPE1/bert/encoder/layer_0/attention/self/query/bias")
    query_kernel = reader.get_tensor("SCOPE1/bert/encoder/layer_0/attention/self/query/kernel")
    beta = reader.get_tensor("SCOPE1/bert/embeddings/layer_normalization/beta")
    gamma = reader.get_tensor("SCOPE1/bert/embeddings/layer_normalization/gamma")
    num_attention_head = 12

    def transform(embedding_like):
        n_voca, hidden_dim = get_shape_list_no_name(embedding_like)
        embedding_like = tf.cast(embedding_like, tf.float32)
        out = apply_layer_norm(embedding_like, gamma, beta, 1e-3)
        out = tf.matmul(out, query_kernel) + query_bias
        out = tf.reshape(out, [n_voca, num_attention_head, -1])
        return out

    print(qtype_embedding.dtype)
    print(word_embedding.dtype)
    # word_embedding_out = transform(word_embedding)
    # qtype_embedding_out = transform(qtype_embedding)
    similarity_matrix = tf.matmul(qtype_embedding, word_embedding, transpose_b=True)
    tokenizer = get_tokenizer()
    for j in range(len(qtype_embedding)):
        score_for_word = similarity_matrix[j]
        ranked = tf.argsort(score_for_word)[::-1]

        s_list = []
        for token_id in ranked[:10]:
            token_id = int(token_id.numpy())
            token = tokenizer.inv_vocab[token_id]
            s = "{0}({1:.2f})".format(token, score_for_word[token_id])
            s_list.append(s)
        s_concat = " ".join(s_list)

        print("Cluster {}: ".format(j) + s_concat)


if __name__ == "__main__":
    main()