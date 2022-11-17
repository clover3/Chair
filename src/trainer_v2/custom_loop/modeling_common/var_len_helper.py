import tensorflow as tf

from data_generator.special_tokens import CLS_ID, SEP_ID


def concat(seg1, seg2):
    CLS = [CLS_ID]
    SEP = [SEP_ID]
    input_ids = tf.concat([CLS, seg1, SEP, seg2, SEP], axis=0)
    seg0_s = tf.zeros_like(seg1, tf.int32)
    seg1_s = tf.ones_like(seg2, tf.int32)
    segment_ids = tf.concat([[0], seg0_s, [0], seg1_s, [1]], axis=0)
    return input_ids, segment_ids


def get_pad_fn(max_seq_length):
    def do_pad(v):
        v = v[:max_seq_length]
        padding_len = max_seq_length - tf.shape(v)[-1]
        padding = [[0, padding_len]]
        padded = tf.pad(v, padding, 'CONSTANT', constant_values=0)
        return padded
    return do_pad
