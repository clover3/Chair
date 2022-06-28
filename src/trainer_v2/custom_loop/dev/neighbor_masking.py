import tensorflow as tf

from trainer_v2.custom_loop.modeling_common.network_utils import int_or, mask_shift_repeat, build_chunk_attention_mask, \
    ChunkAttentionMaskLayer, ChunkAttentionMaskLayerFreeP


def main():
    B_rule = tf.constant([0, 1, 1, 0, 1, 0])
    m_a0 = tf.constant([
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    m_goal = tf.constant([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    m_i = mask_shift_repeat(B_rule)
    assert tf.reduce_sum(m_i - m_goal) == 0


def main2():
    seg1_len = 10
    seg2_len = 6
    chunk_st_mask_p = [0] * seg1_len
    chunk_st_mask_h = [1, 0, 0, 1, 0, 1]
    segment_ids = [0] * seg1_len + [1] * seg2_len
    chunk_st_mask = chunk_st_mask_p + chunk_st_mask_h
    assert len(chunk_st_mask) == seg1_len + seg2_len

    chunk_st_mask_batch = tf.expand_dims(chunk_st_mask, axis=0)
    mask = build_chunk_attention_mask(chunk_st_mask_batch)
    print(mask.shape)
    mask = tf.reduce_sum(mask, axis=0)
    is_seg1 = tf.cast(tf.equal(segment_ids, 0), tf.int32)
    mask_p_to_h = tf.tile(tf.expand_dims(is_seg1, axis=0), [seg1_len + seg2_len, 1])
    # H can see P but not vice versa
    print(int_or(mask, mask_p_to_h))
    # mask0_0 = tf.ones([seg1_len, seg1_len], tf.int32)
    # mask0_1 = tf.ones([seg1_len, seg2_len], tf.int32)
    # mask1_0 = tf.zeros([seg2_len, seg1_len], tf.int32)
    # mask1_1 = build_seg_aftention_mask(chunk_st_mask)
    # mask = tf.concat(
    #     [
    #         tf.concat([mask0_0, mask0_1], axis=1),
    #         tf.concat([mask1_0, mask1_1], axis=1),
    #     ], axis=0
    # )
    # print(mask)


def main3():
    p_array = tf.ones([2, 30], tf.int32)
    h_array = tf.ones([2, 30], tf.int32)
    attention_mask = ChunkAttentionMaskLayerFreeP()([p_array, h_array])
    for row in attention_mask[0][:60]:
        print(row.numpy().tolist())


def main4():
    segment_ids = tf.zeros([2, 30], tf.int32)
    out_mask = ChunkAttentionMaskLayer()(segment_ids)[0]

    for i in range(30):
        print(out_mask[i][:30].numpy().tolist())


if __name__ == "__main__":
    main4()