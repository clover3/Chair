import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, set_level_debug
from tlm.model.base import BertModel
from tlm.model_cnfig import JsonConfig
from tlm.robust.windowed_predictor import model_fn_classification
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.input_fn import input_fn_query_doc
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    set_level_debug()
    tf_logging.info("Run windowed_predictor")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    is_training = FLAGS.do_train
    input_files = get_input_files_from_flags(FLAGS)
    input_fn = input_fn_query_doc(input_files, FLAGS, is_training)
    model_fn = model_fn_classification(config, train_config, BertModel,
                                       FLAGS.max_seq_length, FLAGS.max_query_len, FLAGS.max_doc_len)
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()