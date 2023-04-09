from trainer_v2.per_project.transparency.mmp.tt_model.tt1 import ScoringLayer
import numpy as np
import tensorflow as tf


def get_null_mask(input_ids):
    all_zero = tf.reduce_all(input_ids == 0, axis=2)
    m = tf.cast(tf.logical_not(all_zero), tf.float32)
    return tf.expand_dims(m, axis=2)


def main():
    hidden_size = 10
    num_term = 10
    batch_size = 2
    sl = ScoringLayer()
    num_window = num_term
    max_token_per_term = 4
    padding = [0] * max_token_per_term
    term1 = [1, 2, 3, 4]
    term2 = [3, 0, 0, 0]
    term3 = [3, 1, 0, 0]
    term4 = [3, 1, 2, 0]

    def tensor(l):
        return tf.constant(l)

    def check_exact_match(q_input_ids, d_input_ids):
        q_repeat = tf.tile(tf.expand_dims(q_input_ids, axis=2), [1, 1, num_window, 1])  # [B, M, M, W]
        d_repeat = tf.tile(tf.expand_dims(d_input_ids, axis=1), [1, num_window, 1, 1])
        em = tf.reduce_all(tf.equal(q_repeat, d_repeat), axis=3)  # [B, M, M]
        return tf.cast(em, tf.float32)

    # [B, M]
    def get_expanded_doc_tf(d_expanded_tf_rel, q_bow, d_bow):
        em_f = check_exact_match(q_bow['input_ids'], d_bow['input_ids'])  # exact match as float (0.0 or 1.0)
        d_tf = d_expanded_tf_rel + em_f
        tf_multiplier = tf.tile(tf.expand_dims(d_bow['tfs'], axis=1), [1, num_window, 1])
        tf_multiplier = tf.cast(tf_multiplier, tf.float32)
        expanded_term_df = tf.reduce_sum(d_tf * tf_multiplier, axis=2)
        return expanded_term_df

    q_rep = np.zeros([batch_size, num_term, hidden_size])
    q_rep[1, 1, 0] = 2
    d_rep = np.zeros([batch_size, num_term, hidden_size])
    d_rep[1, 1, 0] = 1
    q_bow = {
        'input_ids': tensor([
            [term1, term2] + [padding] * 8,
            [term1, term2] + [padding] * 8,
        ]),
        'tfs':  tensor([[1, 1] + [0] * 8] * 2)
    }
    d_bow = {
        'input_ids': tensor([
            [term1, term3] + [padding] * 8,
            [term1, term4] + [padding] * 8,
        ]),
        'tfs':  tensor([[1, 2] + [0] * 8] * 2)
    }
    q_rep = get_null_mask(q_bow['input_ids']) * q_rep
    d_rep = get_null_mask(d_bow['input_ids']) * d_rep
    d_t = tf.transpose(d_rep, [0, 2, 1])
    d_expanded_tf = tf.matmul(q_rep, d_t)  # [B, M, M]
    print("d_expanded_tf", d_expanded_tf)
    d_tf = get_expanded_doc_tf(d_expanded_tf, q_bow, d_bow)  # [B, M]
    print("d_tf", d_tf)


    output = sl([q_rep, d_rep, q_bow, d_bow])
    print("output", output)


if __name__ == "__main__":
    main()