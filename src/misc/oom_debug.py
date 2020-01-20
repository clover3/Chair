import tensorflow as tf

from models.transformer.optimization_v2 import create_optimizer
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.training.input_fn import input_fn_builder_unmasked
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


class OomReportingHook(tf.estimator.SessionRunHook):
    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs(fetches=[],  # no extra fetches
                      options=tf.compat.v1.RunOptions(
                      report_tensor_allocations_upon_oom=True))



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


        initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.02)

        embedding_table = tf.compat.v1.get_variable(
            name="embedding",
            shape=[100, 128, 1024],
            initializer=initializer)
        t = embedding_table
        for i in range(20):
            t = tf.reshape(t, [10, 1280, -1])
            t = tf.keras.layers.Dense(1024, kernel_initializer=initializer)(t)
            t = tf.reshape(t, [100, 128, -1])
            t = tf.nn.dropout(t, rate=0.5)
        t = tf.keras.layers.Dense(1024, kernel_initializer=initializer)(t)
        t = tf.nn.dropout(t, rate=0.5)

        total_loss = tf.reduce_mean(t)
        train_op = create_optimizer(total_loss, 1e-4, 1000, 1000, True)

        scaffold_fn = None
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                train_op=train_op,
                loss=total_loss,
                training_hooks=[OomReportingHook()],
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
