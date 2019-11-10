
from tlm.model.masking import *


def test_routine():
    input_ids = tf.ones([100 ,256], dtype=tf.dtypes.int32)
    input_mask = tf.ones([100 ,256], dtype=tf.dtypes.int32)
    segment_ids = tf.ones([100 ,256], dtype=tf.dtypes.int32)


    n_sample = 30
    mask_token = 3
    masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
        = sample(input_ids, segment_ids, n_sample, mask_token)

    print("mask_input_ids: ", masked_input_ids.shape)
    print("masked_lm_positions: ", masked_lm_positions.shape)
    print("masked_lm_ids: ", masked_lm_ids.shape)




if __name__ == "__main__":
    test_routine()