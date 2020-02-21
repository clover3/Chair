from my_tf import tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.topic_vector import TopicVectorBert
from tlm.model_cnfig import JsonConfig
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.input_fn_common import format_dataset
from tlm.training.lm_model_fn import model_fn_lm
from tlm.training.train_config import LMTrainConfig
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


def input_fn_topic_fn(input_files,
                              flags,
                              is_training,
                              num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length

        name_to_features = {
                "input_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids":tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "topic_ids":tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


@report_run
def main(_):
    tf_logging.info("Train topic_vector")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = LMTrainConfig.from_flags(FLAGS)
    is_training = FLAGS.do_train
    input_files = get_input_files_from_flags(FLAGS)
    input_fn = input_fn_topic_fn(input_files, FLAGS, is_training)
    model_fn = model_fn_lm(config, train_config, TopicVectorBert.factory, get_masked_lm_output, True)
    return run_estimator(model_fn, input_fn)

if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
