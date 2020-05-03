import tensorflow as tf
import tensorflow_probability as tfp

from models.transformer.bert_common_v2 import get_shape_list2


def candidate_gen(input_ids, input_mask, segment_ids, n_trial):
    seed = 0

    # draw random interval

    batch_size, input_len = get_shape_list2(input_ids)
    indice = draw_starting_point(batch_size, input_len, input_mask, n_trial, seed)
    flat_indice = tf.reshape(indice, [batch_size*n_trial]) # [ batch_size, m]

    geo = tfp.distributions.Geometric([0.5])
    length_arr = tf.squeeze(tf.cast(geo.sample(indice.shape) + 1, tf.int32), 2)

    length_arr_flat = tf.reshape(length_arr, [-1])

    new_input_ids = drop_middle(batch_size, flat_indice, input_ids, input_len, length_arr_flat, n_trial)
    new_segment_ids = drop_middle(batch_size, flat_indice, segment_ids, input_len, length_arr_flat, n_trial)
    new_input_mask = drop_middle(batch_size, flat_indice, input_mask, input_len, length_arr_flat, n_trial)

    return new_input_ids, new_segment_ids, new_input_mask, indice, length_arr


def delete_one_by_one(input_ids, input_mask, segment_ids, n_trial):
    NotImplemented


def add_with_cap(a, b, max_len):
    r = a + b
    r_cap = tf.ones_like(a, tf.int32) * max_len
    return tf.minimum(r, r_cap)


def drop_middle(batch_size, flat_indice, input_arr, input_len, drop_length, m):
    start_2 = add_with_cap(flat_indice, drop_length, input_len)
    repeat_input_ids = tf.reshape(tf.tile(input_arr, [1, m]), [batch_size * m, input_len])
    split_indice = translate_split_index(batch_size, flat_indice, input_len, m)

    flat_repeat_input_ids = tf.reshape(repeat_input_ids, [-1])

    head_and_dummy = tf.RaggedTensor.from_row_splits(values=flat_repeat_input_ids,
                                           row_splits=split_indice)
    head = tf.gather(head_and_dummy, tf.range(m * batch_size) * 2)
    split_indice = translate_split_index(batch_size, start_2, input_len, m)
    dummy_and_tail = tf.RaggedTensor.from_row_splits(flat_repeat_input_ids, split_indice)
    tail = tf.gather(dummy_and_tail, tf.range(m * batch_size) * 2 + 1)

    new_input = tf.concat([head, tail], axis=1)
    return new_input.to_tensor(default_value=0, shape=[batch_size * m, input_len])


def translate_split_index(batch_size, flat_indice, input_len, m):
    end_sent = tf.ones_like(flat_indice, tf.int32) * input_len
    i2 = tf.transpose(tf.stack([flat_indice, end_sent]), [1, 0])
    shift = tf.expand_dims(tf.range(m * batch_size) * input_len, 1)
    i2 = i2 + shift
    split_indice = tf.concat([[0], tf.reshape(i2, [-1])], axis=0)
    # [m*batch_size, seq_len]
    return split_indice


def draw_starting_point(batch_size, input_len, input_mask, m, seed):
    rand = tf.random.uniform(
        [batch_size, input_len],
        minval=0,
        maxval=1,
        dtype=tf.dtypes.float32,
        seed=seed,
        name="masking_uniform"
    )
    rand = rand * tf.cast(input_mask, tf.float32)
    _, indice = tf.math.top_k(
        rand[:, 1:-1],
        k=m,
        sorted=False,
        name="masking_top_k"
    )
    return indice + 1