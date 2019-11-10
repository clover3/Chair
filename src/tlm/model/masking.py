from tlm.model.base import BertModel
from models.transformer.bert_common_v2 import get_shape_list
import tensorflow as tf

def pad_as_shape(value, shape_like, dims):
    for _ in range(dims):
        value = tf.expand_dims(value, 0)
    value = tf.ones_like(shape_like) * value
    return value

def get_shape_list2(tensor):
    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(input=tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

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


def sample(input_ids, input_masks, n_sample, mask_token):

    rand = tf.random.uniform(
        input_ids.shape,
        minval=0,
        maxval=1,
        dtype=tf.dtypes.float32,
        seed=None,
        name=None
    )
    rand = rand * tf.cast(input_masks, tf.float32)

    _, indice = tf.math.top_k(
        rand,
        k=n_sample,
        sorted=False,
        name=None
    )
    masked_lm_positions = indice # [batch, n_samples]
    masked_lm_ids = tf.gather(input_ids, masked_lm_positions, axis=-1, batch_dims=0)
    masked_lm_weights = tf.ones_like(masked_lm_positions)
    masked_input_ids = scatter_with_batch(input_ids, indice, mask_token)
    return masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights
