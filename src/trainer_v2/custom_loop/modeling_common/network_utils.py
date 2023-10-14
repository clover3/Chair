import tensorflow as tf

from models.transformer.bert_common_v2 import get_shape_list2


def vector_three_feature(v1, v2):
    concat_layer = tf.keras.layers.Concatenate()
    concat = concat_layer([v1, v2])
    sub = v1 - v2
    dot = tf.multiply(v1, v2)
    output = tf.concat([sub, dot, concat], axis=-1, name="three_feature")
    return output


class VectorThreeFeature(tf.keras.layers.Layer):
    def __init__(self):
        super(VectorThreeFeature, self).__init__()

    def call(self, inputs, *args, **kwargs):
        v1, v2 = inputs
        return vector_three_feature(v1, v2)


class MeanProjection(tf.keras.layers.Layer):
    def __init__(self, project_dim, prefix):
        super(MeanProjection, self).__init__()
        Dense = tf.keras.layers.Dense
        self.projector: tf.keras.layers.Dense = Dense(project_dim, activation='relu', name="{}/project".format(prefix))

    def call(self, inputs, *args, **kwargs):
        seq_p = self.projector(inputs)
        seq_m = tf.reduce_mean(seq_p, axis=1)
        return seq_m


def split_stack_input(input_ids_like_list,
                total_seq_length: int,
                window_length: int,
                ):
    # e.g. input_id_like_list[0] shape is [8, 250 * 4],  it return [8 * 4, 250]
    num_window = int(total_seq_length / window_length)
    batch_size, _ = get_shape_list2(input_ids_like_list[0])

    def r2to3(arr):
        return tf.reshape(arr, [batch_size, num_window, window_length])

    return list(map(r2to3, input_ids_like_list))


class SplitSegmentIDLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SplitSegmentIDLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        rep_middle, l_input_ids, token_type_ids = inputs
        # rep_middle: [batch_size, seq_length, hidden_dim]

        def slice_segment_pad_value(segment_id_val):
            is_target_seg_mask = tf.logical_and(tf.equal(token_type_ids, segment_id_val), tf.not_equal(l_input_ids, 0))
            is_target_seg_mask = tf.cast(tf.expand_dims(is_target_seg_mask, 2), tf.float32)
            rep_middle_masked = tf.multiply(rep_middle, is_target_seg_mask)
            return rep_middle_masked

        rep_middle0 = slice_segment_pad_value(0)
        rep_middle1 = slice_segment_pad_value(1)

        return rep_middle0, rep_middle1


class SplitSegmentIDLayerWVar(tf.keras.layers.Layer):
    def __init__(self, hidden_dims):
        super(SplitSegmentIDLayerWVar, self).__init__()
        init = tf.random_normal_initializer()
        self.empty_embedding = tf.Variable(
            initial_value=init(shape=(hidden_dims,), dtype="float32"), trainable=True
        )

    def call(self, inputs, *args, **kwargs):
        rep_middle, l_input_ids, token_type_ids = inputs
        # rep_middle: [batch_size, seq_length, hidden_dim]
        batch_size, seq_length = get_shape_list2(l_input_ids)
        empty_embedding_seq = tf_tile_after_expand_dims(self.empty_embedding, [0, 1], [batch_size, seq_length, 1])

        def slice_segment_pad_value(segment_id_val):
            input_mask = tf.not_equal(l_input_ids, 0)
            is_target_seg_mask = tf.logical_and(tf.equal(token_type_ids, segment_id_val), input_mask)
            is_target_seg_mask = tf.cast(tf.expand_dims(is_target_seg_mask, 2), tf.float32)
            rep_middle_masked = tf.multiply(rep_middle, is_target_seg_mask)
            is_not_target_seq_mask = (1.0 - is_target_seg_mask)
            empty_embedding_seq_masked = tf.multiply(empty_embedding_seq, is_not_target_seq_mask)
            return rep_middle_masked + empty_embedding_seq_masked

        rep_middle0 = slice_segment_pad_value(0)
        rep_middle1 = slice_segment_pad_value(1)

        return rep_middle0, rep_middle1


class TwoLayerDense(tf.keras.layers.Layer):
    def __init__(self, hidden_size, hidden_size2,
                 activation1='relu', activation2=tf.nn.softmax,
                 **kwargs
                 ):
        super(TwoLayerDense, self).__init__(**kwargs)
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation=activation1)
        self.layer2 = tf.keras.layers.Dense(hidden_size2, activation=activation2)

    def call(self, inputs, *args, **kwargs):
        hidden = self.layer1(inputs)
        return self.layer2(hidden)


def tf_tile_after_expand_dims(v, expand_dim_list, tile_param):
    v_ex = v
    for expand_dim in expand_dim_list:
        v_ex = tf.expand_dims(v_ex, expand_dim)
    return tf.tile(v_ex, tile_param)


class TileAfterExpandDims(tf.keras.layers.Layer):
    def __init__(self, expand_dim_raw, tile_param):
        super(TileAfterExpandDims, self).__init__()
        if type(expand_dim_raw) == int:
            self.expand_dim_list = [expand_dim_raw]
        else:
            self.expand_dim_list = expand_dim_raw
        self.tile_param = tile_param

    def call(self, inputs, *args, **kwargs):
        return tf_tile_after_expand_dims(inputs, self.expand_dim_list, self.tile_param)



class SplitSegmentIDWMeanLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SplitSegmentIDWMeanLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        rep_middle, l_input_ids, token_type_ids = inputs
        # rep_middle: [batch_size, seq_length, hidden_dim]

        def slice_segment_w_mean(segment_id_val):
            is_target_seg_mask = tf.logical_and(tf.equal(token_type_ids, segment_id_val), tf.not_equal(l_input_ids, 0))
            is_target_seg_mask = tf.expand_dims(is_target_seg_mask, 2)
            is_target_seg_mask = tf.cast(is_target_seg_mask, tf.float32)
            rep_middle_masked = tf.multiply(rep_middle, is_target_seg_mask)
            eps = 1e-6
            n_tokens = tf.reduce_sum(is_target_seg_mask, axis=1) + eps  # [batch_size, 1]
            summed = tf.reduce_sum(rep_middle_masked, axis=1) # [batch_size, project_dim]
            return tf.divide(summed, n_tokens)

        rep_middle0 = slice_segment_w_mean(0)
        rep_middle1 = slice_segment_w_mean(1)

        return rep_middle0, rep_middle1


class SplitSegmentID(tf.keras.layers.Layer):
    def __init__(self):
        super(SplitSegmentID, self).__init__()

    def call(self, inputs, *args, **kwargs):
        l_input_ids, token_type_ids = inputs

        def slice_segment_w_mean(segment_id_val):
            is_target_seg_mask = tf.logical_and(tf.equal(token_type_ids, segment_id_val), tf.not_equal(l_input_ids, 0))
            is_target_seg_mask = tf.cast(is_target_seg_mask, tf.int32)
            masked_input_ids = tf.multiply(l_input_ids, is_target_seg_mask)
            return masked_input_ids

        input_ids1 = slice_segment_w_mean(0)
        input_ids2 = slice_segment_w_mean(1)

        n_shift = tf.reduce_sum(tf.cast(tf.not_equal(input_ids1, 0), tf.int32), axis=1, keepdims=True)
        input_ids2_n_shift = tf.concat([input_ids2, -n_shift], axis=1)
        input_ids2_r = tf.map_fn(fn=lambda x: tf.roll(x[:-1], shift=x[-1], axis=0),
                                   elems=input_ids2_n_shift)

        return input_ids1, input_ids2_r


def logical_to_int(a, b, logical_fn):
    out_b = logical_fn(tf.cast(a, tf.bool), tf.cast(b, tf.bool))
    return tf.cast(out_b, tf.int32)


def int_or(a, b):
    return logical_to_int(a, b, tf.logical_or)


def int_and(a, b):
    return logical_to_int(a, b, tf.logical_and)


def mask_shift_repeat_old(is_continue_chunk):
    B, L = get_shape_list2(is_continue_chunk)
    n_shift = 10
    m_b_r1 = tf.tile(tf.expand_dims(is_continue_chunk, -1), [1, 1, L])
    m_b_r0 = tf.transpose(m_b_r1, [0, 2, 1])
    identity_2d = tf.eye(L, dtype=tf.int32)
    identity_3d = tf.expand_dims(identity_2d, 0)
    m_0 = tf.tile(identity_3d, [B, 1, 1])

    def shift_dim0_down(t, step):
        t_s = tf.concat([tf.zeros([B, step, L], tf.int32), t[:, :-step]], axis=1)
        return t_s

    def shift_dim1_down(t, step):
        t_s = tf.concat([tf.zeros([B, L, step], tf.int32), t[:, :, :-step]], axis=2)
        return t_s

    m_i = m_0
    for _ in range(n_shift):
        m_i_prev = m_i
        # M_2[a, b] = M_1[a, b-1] and B[b]  or M_1[a, b]
        #  m_b_r0[i1, j] == m_b_r0[i2, j]
        m_shift_by0 = int_and(shift_dim0_down(m_i_prev, 1), m_b_r1)
        m_shift_by1 = int_and(shift_dim1_down(m_i_prev, 1), m_b_r0)
        m_i = int_or(int_or(m_shift_by1, m_i_prev), m_shift_by0)
    return m_i


def mask_shift_repeat(is_continue_chunk):
    B, L = get_shape_list2(is_continue_chunk)
    n_shift = 10
    m_b_r1 = tf.tile(tf.expand_dims(is_continue_chunk, -1), [1, 1, L])
    m_b_r0 = tf.transpose(m_b_r1, [0, 2, 1])
    identity_2d = tf.eye(L, dtype=tf.int32)
    identity_3d = tf.expand_dims(identity_2d, 0)
    m_0 = tf.tile(identity_3d, [B, 1, 1])
    horizontal_zero = tf.zeros([B, 1, L], tf.int32)
    def shift_dim0_down(t):
        # return tf.roll(t, axis=1, shift=1)
        t_s = tf.concat([horizontal_zero, t[:, :-1]], axis=1)
        return t_s
        # return t_s

    m_i = m_0
    for _ in range(n_shift):
        m_i_prev = m_i
        # M_2[a, b] = M_1[a, b-1] and B[b]  or M_1[a, b]
        #  m_b_r0[i1, j] == m_b_r0[i2, j]
        m_shift_by0 = tf.multiply(shift_dim0_down(m_i_prev), m_b_r1)
        m_i = tf.maximum(m_i_prev, m_shift_by0)

    out_m = tf.maximum(tf.transpose(m_i, [0, 2, 1]), m_i)
    out_m = tf.cast(out_m, tf.int32)

    return out_m


def build_chunk_attention_mask(chunk_st_mask):
    B_rule = 1 - chunk_st_mask
    m_i = mask_shift_repeat(B_rule)
    return m_i


class ChunkAttentionMaskLayer(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return build_chunk_attention_mask(inputs)


class ChunkAttentionMaskLayerFreeP(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        p_array, h_array = inputs
        B, L1 = get_shape_list2(p_array)
        B, L2 = get_shape_list2(h_array)
        mask00 = tf.ones([B, L1, L1], tf.int32)
        mask01 = tf.zeros([B, L1, L2], tf.int32)
        mask10 = tf.ones([B, L2, L1], tf.int32)
        mask11 = build_chunk_attention_mask(h_array)
        output_mak = tf.concat([
            tf.concat([mask00, mask01], axis=2),
            tf.concat([mask10, mask11], axis=2)
        ], axis=1)
        return output_mak
