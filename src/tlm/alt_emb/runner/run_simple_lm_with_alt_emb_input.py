
from my_tf import tf
from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model_cnfig import JsonConfig
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.input_fn import input_fn_builder_unmasked_alt_emb2
from tlm.training.lm_model_fn import model_fn_lm
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator

@report_run
def main(_):
    tf_logging.info("Train MLM  with BERT like")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)

    is_training = FLAGS.do_train
    input_files = get_input_files_from_flags(FLAGS)

    input_fn = input_fn_builder_unmasked_alt_emb2(input_files, FLAGS, is_training)

    model_fn = model_fn_lm(config, train_config, BertModel, get_masked_lm_output)
    run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
