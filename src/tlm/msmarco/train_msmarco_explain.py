import warnings

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.tlm.explain_model_fn import model_fn_explain
from tlm.training import flags_wrapper
from tlm.training.train_config import TrainConfigEx

warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
import tlm.model.base as modeling
from trainer.tpu_estimator import run_estimator
from tlm.training.flags_wrapper import show_input_files
from tlm.training.train_flags import *
from tlm.msmarco.msmarco_input_fn import input_fn_builder


@report_run
def main(_):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    input_files = flags_wrapper.get_input_files()
    train_config = TrainConfigEx.from_flags(FLAGS)
    show_input_files(input_files)
    model_fn = model_fn_explain(
        bert_config=bert_config,
        train_config=train_config,
        logging=tf_logging,
    )
    is_training = FLAGS.do_train
    input_fn = input_fn_builder(input_files, FLAGS.max_seq_length, is_training)

    r = run_estimator(model_fn, input_fn)
    return r


if __name__ == "__main__":
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
