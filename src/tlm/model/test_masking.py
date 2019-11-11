
from tlm.model.masking import *
from data_generator import special_tokens

def test_routine():
    input_ids = tf.constant([[special_tokens.CLS_ID, 1002,1003,1004,1005, special_tokens.SEP_ID, special_tokens.PAD_ID],
                             [special_tokens.CLS_ID, 1005, 1002, 1001, 1004, special_tokens.SEP_ID,
                              special_tokens.PAD_ID]
                             ], dtype=tf.dtypes.int32)
    input_mask = tf.ones([2 ,7], dtype=tf.dtypes.int32)
    segment_ids = tf.ones([2, 7], dtype=tf.dtypes.int32)


    n_sample = 3
    mask_token = 3
    masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
        = do_masking(input_ids, segment_ids, n_sample, mask_token)

    print("mask_input_ids: ", masked_input_ids.shape)
    print("masked_lm_positions: ", masked_lm_positions.shape)
    print("masked_lm_ids: ", masked_lm_ids.shape)

    print(masked_input_ids)
    print(masked_lm_positions)




if __name__ == "__main__":
    test_routine()