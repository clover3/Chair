import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model_cnfig import JsonConfig
from tlm.tlm.lm_nli_shared_model_fn import model_fn_nli_lm, input_fn_builder, decay_combine, \
    AddLayerSharingModel
from tlm.training import flags_wrapper
from tlm.training.train_config import LMTrainConfig
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Train nli_lm_shared")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = LMTrainConfig.from_flags(FLAGS)

    is_training = FLAGS.do_train
    input_files = flags_wrapper.get_input_files()

    input_fn = input_fn_builder(input_files, FLAGS, is_training, True)
    model_fn = model_fn_nli_lm(config, train_config, AddLayerSharingModel, decay_combine)
    run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
