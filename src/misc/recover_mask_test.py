import os

import tensorflow as tf

from tlm.tlm.bfn_loss_predict_gen import PredictionOutput


def recover_mask(input_ids, masked_lm_positions, masked_lm_ids):
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    flat_input_ids = tf.reshape(input_ids, [-1])
    offsets = tf.range(batch_size) * seq_length
    masked_lm_positions = masked_lm_positions + tf.expand_dims(offsets, 1)
    masked_lm_positions = tf.reshape(masked_lm_positions, [-1, 1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])

    output = tf.tensor_scatter_nd_update(flat_input_ids, masked_lm_positions, masked_lm_ids)
    output = tf.reshape(output, [batch_size, seq_length])
    output = tf.concat([input_ids[:, :1], output[:,1:]], axis=1)
    return output


def get_data():
    name1 = os.path.join("disk_output", "bert_{}.pickle".format("815"))
    output1 = PredictionOutput(name1)
    return output1


def main():
    data = get_data()
    output = recover_mask(data.input_ids[:10], data.masked_lm_positions[:10], data.masked_lm_ids[:10])

    for i in range(10):
        print(data.input_ids[i])
        print(output[i])



main()