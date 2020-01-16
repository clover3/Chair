import tensorflow as tf

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


        t = input_ids + input_mask + segment_ids
        t = t * next_sentence_labels
        total_loss = tf.reduce_sum(tf.cast(t[:,0], tf.float32))

        scaffold_fn = None
        eval_metrics = (lambda x:x, [
                input_ids,
        ])
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
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
