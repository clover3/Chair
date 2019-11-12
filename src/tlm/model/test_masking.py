
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
        = random_masking(input_ids, segment_ids, n_sample, mask_token)
    print(input_ids)
    print("mask_input_ids: ", masked_input_ids.shape)
    print(masked_input_ids)
    print("masked_lm_positions: ", masked_lm_positions.shape)
    print(masked_lm_positions)
    print("masked_lm_ids: ", masked_lm_ids.shape)
    print(masked_lm_ids)


def test_routine2():
    input_ids = tf.constant([[special_tokens.CLS_ID, 1002, 1003, 1004, 1005, special_tokens.SEP_ID, special_tokens.PAD_ID],
                             [special_tokens.CLS_ID, 1005, 1002, 1001, 1004, special_tokens.SEP_ID,special_tokens.PAD_ID]
                             ], dtype=tf.dtypes.int32)
    input_mask = tf.ones([2 ,7], dtype=tf.dtypes.int32)
    segment_ids = tf.ones([2, 7], dtype=tf.dtypes.int32)
    priority_score = tf.constant([[0.1, 0.5, 0.4, -0.5, 0.1, 0.1, 0.1],
                                  [0.1, 0.5, 0.4, 0.01, 0.1, 0.1, 0.1]])
    prob = tf.nn.softmax(priority_score, axis=1)
    print("prob")
    print(prob)
    n_sample = 3
    mask_token = 3
    alpha = 0.1
    masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
        = biased_masking(input_ids, input_mask, priority_score, alpha, n_sample, mask_token)
    print(input_ids)
    print("mask_input_ids: ", masked_input_ids.shape)
    print(masked_input_ids)
    print("masked_lm_positions: ", masked_lm_positions.shape)
    print(masked_lm_positions)
    print("masked_lm_ids: ", masked_lm_ids.shape)
    print(masked_lm_ids)




if __name__ == "__main__":
    test_routine2()