


import tensorflow as tf

from tlm.explain_payload_gen import candidate_gen
from tlm.tlm.explain_model_fn import get_mask, get_informative


def select_best(best_run, #[batch_size, num_class]
                indice, # [batch_size, n_trial]
                length_arr):

    best_run_ex = tf.expand_dims(best_run, 2)
    d = tf.zeros_like(best_run_ex, tf.int32)
    best_run_idx = tf.concat([d, best_run_ex], axis=2)

    def select(arr):
        arr_ex = tf.expand_dims(arr, 1)
        return tf.gather_nd(arr_ex, best_run_idx, batch_dims=1)

    good_deletion_idx = select(indice)
    good_deletion_length = select(length_arr)
    return good_deletion_idx, good_deletion_length


def main():
    tf.random.set_seed(13)
    input_ids = tf.constant([[1,2,3,4,5,6,7,8,9,0],
                 [11, 12, 13, 14, 15, 6, 7, 8, 0, 0]])
    input_mask = tf.constant([[1,1,1,1,1,1,1,1,1,0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
    segment_ids = tf.constant([[1,1,1,1,1,1,1,1,1,0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

    print(input_ids.shape)

    new_input_ids, new_segment_ids, new_input_mask, indice, length_arr = \
        candidate_gen(input_ids, input_mask, segment_ids, 3)

    print("new_input_ids", new_input_ids)
    print("new_segment_ids", new_segment_ids)
    print("new_input_mask", new_input_mask)
    print("indice", indice)
    print("length_arr", length_arr)

    best_run = tf.constant([[0, 0, 1], [1, 1, 0]])
    good_deletion_idx, good_deletion_length = select_best(best_run, indice, length_arr)
    print("good_deletion_idx", good_deletion_idx)
    mask = get_mask(good_deletion_idx, good_deletion_length, 10)
    print(mask)

def get_informative_test():
    orig_probs = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]
    new_probs = [[[0.9, 0.1, 0.0],
                 [0.5, 0.5, 0.0],
                 [0.5, 0.5, 0.0],
                 [0.5, 0.5, 0.0],
                 [0.1, 0.2, 0.9]],
                [[0.9, 0.1, 0.0],
                 [0.1, 0.2, 0.9],
                 [0.1, 0.2, 0.9],
                 [0.1, 0.2, 0.9],
                 [0.1, 0.2, 0.9]]]
    r = get_informative(new_probs, orig_probs, 0.3)
    print(r)

main()