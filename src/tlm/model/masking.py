import tensorflow as tf

from data_generator import special_tokens
from models.transformer.bert_common_v2 import gather_index2d, get_shape_list2


def pad_as_shape(value, shape_like, dims):
    for _ in range(dims):
        value = tf.expand_dims(value, 0)
    value = tf.ones_like(shape_like) * value
    return value


def scatter_with_batch(input_ids, indice, mask_token):
    batch_size = get_shape_list2(input_ids)[0]
    seq_length = get_shape_list2(input_ids)[1]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    indices = tf.reshape(indice + flat_offsets, [-1, 1])
    tensor = tf.reshape(input_ids, [batch_size*seq_length])

    updates = tf.reshape(tf.ones_like(indices) * mask_token, [-1])
    flat_output = tf.tensor_scatter_nd_update(tensor, indices, updates)
    return tf.reshape(flat_output, [batch_size, seq_length])


def remove_special_mask(input_ids, input_masks, t, special_token_list=None):
    special_mask = tf.logical_or(tf.equal(input_ids, special_tokens.SEP_ID),
                                 tf.equal(input_ids, special_tokens.CLS_ID))

    if special_token_list is None:
        special_token_list = []
    for token in special_token_list:
        special_mask = tf.logical_or(tf.equal(input_ids, token), special_mask)

    special_mask = tf.logical_or(special_mask, tf.equal(input_ids, special_tokens.UNUSED_4))
    t = t * tf.cast(input_masks, tf.float32)
    t = t * tf.cast(tf.logical_not(special_mask), tf.float32)
    return t


def random_masking(input_ids, input_masks, n_sample, mask_token, seed=None, special_tokens=None):
    rand = tf.random.uniform(
        input_ids.shape,
        minval=0,
        maxval=1,
        dtype=tf.dtypes.float32,
        seed=seed,
        name="masking_uniform"
    )

    if special_tokens is None:
        special_tokens = []
    rand = remove_special_mask(input_ids, input_masks, rand, special_tokens)
    _, indice = tf.math.top_k(
        rand,
        k=n_sample,
        sorted=False,
        name="masking_top_k"
    )
    masked_lm_positions = indice # [batch, n_samples]
    #masked_lm_ids = tf.gather(input_ids, masked_lm_positions, axis=1, batch_dims=0)
    masked_lm_ids = gather_index2d(input_ids, masked_lm_positions)
    masked_lm_weights = tf.ones_like(masked_lm_positions, dtype=tf.float32)
    masked_input_ids = scatter_with_batch(input_ids, indice, mask_token)
    return masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights


def biased_masking(input_ids, input_masks, priority_score, alpha, n_sample, mask_token):
    prob = tf.nn.softmax(priority_score, axis=1)
    sequence_shape = get_shape_list2(prob)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]

    # alpha =1 implies uniform masking
    p1 = tf.ones_like(prob, dtype=tf.float32) / seq_length * alpha
    p2 = prob * (1-alpha)
    final_p = p1 + p2

    final_p = remove_special_mask(input_ids, input_masks, final_p)

    indice = tf.random.categorical(
        final_p,
        n_sample,
        dtype=tf.int32,
        seed=None,
        name=None
    )


    masked_lm_positions = indice # [batch, n_samples]
    masked_lm_ids = gather_index2d(input_ids, masked_lm_positions)
    masked_lm_weights = tf.ones_like(masked_lm_positions, dtype=tf.float32)
    masked_input_ids = scatter_with_batch(input_ids, indice, mask_token)
    return masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights



def biased_masking_old(input_ids, input_masks, priority_score, alpha, n_sample, mask_token):
    prob = tf.nn.softmax(priority_score, axis=1)
    sequence_shape = get_shape_list2(prob)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]

    rand = tf.random.uniform(
        prob.shape,
        minval=0,
        maxval=1,
        dtype=tf.dtypes.float32,
        seed=None,
        name=None
    )

    p1 = tf.ones_like(prob, dtype=tf.float32) / seq_length * alpha
    p2 = prob * (1-alpha)
    final_p = p1 + p2

    rand = rand * final_p
    rand = remove_special_mask(input_ids, input_masks, rand)

    _, indice = tf.math.top_k(
        rand,
        k=n_sample,
        sorted=False,
        name=None
    )

    masked_lm_positions = indice # [batch, n_samples]
    masked_lm_ids = gather_index2d(input_ids, masked_lm_positions)
    masked_lm_weights = tf.ones_like(masked_lm_positions, dtype=tf.float32)
    masked_input_ids = scatter_with_batch(input_ids, indice, mask_token)
    return masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights


