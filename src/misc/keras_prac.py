import tensorflow as tf

from models.transformer.bert_common_v2 import get_shape_list


class MyModel:
    def __init__(self):
        pass

    def build(self):
        vocab_size = 40000
        embedding_size = 512
        initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.02)
        embedding_table = tf.compat.v1.get_variable(
            name="embedding",
            shape=[vocab_size, embedding_size],
            initializer=initializer)

        seq_length = 512

        input_ids = tf.keras.layers.Input(shape=(seq_length,))
        input_shape = get_shape_list(input_ids)
        flat_input_ids = tf.reshape(input_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
        output = tf.matmul(one_hot_input_ids, embedding_table)
        output = tf.reshape(output, input_shape + [embedding_size])

        t_list = []
        n_of_t = 10
        for j in range(n_of_t):
            t_list.append(output+j)

        dense = tf.keras.layers.Dense(embedding_size, kernel_initializer=initializer, name="MDense")
        n_of_t = 10

        for i in range(20):
            t = tf.stack(t_list, 0)
            with tf.compat.v1.variable_scope("scope_A", reuse=i > 0):
                t = dense(t)
            t = tf.nn.dropout(t, rate=0.5)

            t_0 = 0
            for j in range(1, n_of_t):
                t_0 += t[j]

            new_t_list = [t_0]
            for j in range(1, n_of_t):
                new_t_list.append(t[j])

            t_list = new_t_list