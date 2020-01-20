import numpy as np
import tensorflow as tf

from models.transformer.optimization_v2 import create_optimizer
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.training.input_fn import input_fn_builder_unmasked
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


def model_fn_builder():
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        next_sentence_labels = features["next_sentence_labels"]

        # t = tf.Variable(np.ones([4,4,128]), tf.float32)
        # initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.02)
        # lstm = tf.keras.layers.LSTM(128, dtype = tf.float32)
        # print(lstm.cell)
        # init_state = tf.zeros([128, 128])
        # whole_seq_output, final_memory_state, final_carry_state = lstm(t, initial_state=init_state)
        #
        inputs = np.random.random([32, 10, 8]).astype(np.float32)


        lstm = tf.compat.v1.keras.layers.LSTM(4, return_sequences=True, return_state=True)

        # whole_sequence_output has shape `[32, 10, 4]`.
        # final_memory_state and final_carry_state both have shape `[32, 4]`.
        whole_sequence_output, final_memory_state, final_carry_state = lstm(inputs)

        total_loss = tf.reduce_sum(final_carry_state)
        train_op = create_optimizer(total_loss, 1e-4, 1000, 1000, True)


        scaffold_fn = None
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                train_op=train_op,
                loss=total_loss,
                scaffold_fn=scaffold_fn)


        return output_spec

    return model_fn



@report_run
def main(_):
    is_training = FLAGS.do_train

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))

    input_fn = input_fn_builder_unmasked(input_files, FLAGS, is_training)
    model_fn = model_fn_builder()

    run_estimator(model_fn, input_fn)

if __name__ == "__main__":
    tf.compat.v1.app.run()
