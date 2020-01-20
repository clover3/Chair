import os

import numpy as np
import tensorflow as tf

import cpath
from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from tf_v2_support import tf1, disable_eager_execution
from tlm.model.horizon import HorizontalAlpha
from tlm.model.lm_objective import get_masked_lm_output_albert
from tlm.model.masking import random_masking
from tlm.model_cnfig import JsonConfig
from tlm.training.train_config import LMTrainConfig
from tlm.training.train_flags import *
from trainer.tf_train_module_v2 import init_session


def define_graph(input_ids, input_mask, segment_ids):
    train_config = LMTrainConfig.from_flags(FLAGS)
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    model = HorizontalAlpha(config, True, False)
    model.call(input_ids, input_mask, segment_ids)
    masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
        = random_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID, 0)

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output_albert(
        config, model.get_sequence_output(), model.get_embedding_table(),
        masked_lm_positions, masked_lm_ids, masked_lm_weights)
    train_op = optimization.create_optimizer_from_config(masked_lm_loss, train_config)

    return train_op



def main(_):
    disable_eager_execution()
    seq_length = 512
    batch_size = FLAGS.train_batch_size
    virtual_input_ids = np.zeros([batch_size, seq_length], np.int)
    with tf.device("/device:gpu:0"):
        input_ids = tf1.placeholder(tf.int32, [batch_size, seq_length])
        input_mask = tf1.placeholder(tf.int32, [batch_size, seq_length])
        segment_ids = tf1.placeholder(tf.int32, [batch_size, seq_length])
        print("Defining grpah...")
        train_op = define_graph(input_ids, input_mask, segment_ids)

    print("Initializing variables...")
    config = tf.compat.v1.ConfigProto(log_device_placement=False,
                                      allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    run_options = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE,
                                 )
    run_metadata = tf1.RunMetadata()
    print("Now running...")
    for i in range(1000):
        _ = sess.run([train_op],
                 feed_dict={
                     input_ids: virtual_input_ids,
                     input_mask: virtual_input_ids,
                     segment_ids: virtual_input_ids,
                },
                  )
        print("step", i)
        i = 0

def boring():
    disable_eager_execution()
    seq_length = 512
    batch_size = 3
    virtual_input_ids = np.zeros([batch_size, seq_length], np.int)

    input_ids = tf1.placeholder(tf.int32, [batch_size, seq_length])
    input_mask = tf1.placeholder(tf.int32, [batch_size, seq_length])
    segment_ids = tf1.placeholder(tf.int32, [batch_size, seq_length])

    train_op = define_graph(input_ids, input_mask, segment_ids)
    tf.compat.v1.summary.scalar('accuracy', 0)
    merged = tf1.summary.merge_all()

    sess = init_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    run_options = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE)
    run_metadata = tf1.RunMetadata()
    train_writer = tf1.summary.FileWriter(os.path.join(cpath.output_path, "horizon_summary"),
                                              sess.graph)

    _, summary_out = sess.run([train_op, merged],
             feed_dict={
                 input_ids: virtual_input_ids,
                 input_mask: virtual_input_ids,
                 segment_ids: virtual_input_ids,
            },
                options = run_options,
              run_metadata = run_metadata)
    i = 0
    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    train_writer.add_summary(summary_out, i)


if __name__ == "__main__":
    tf.compat.v1.app.run()
