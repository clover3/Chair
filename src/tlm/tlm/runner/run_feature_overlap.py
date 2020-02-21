import tensorflow as tf

from taskman_client.wrapper import report_run
from tlm.model_cnfig import JsonConfig
from tlm.tlm.lm_nli_shared_model_fn import SharingFetchGradModel, \
    model_fn_share_fetch_grad
from tlm.training import flags_wrapper
from tlm.training.input_fn import input_fn_builder_unmasked
from tlm.training.train_config import LMTrainConfig
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = LMTrainConfig.from_flags(FLAGS)

    input_files = flags_wrapper.get_input_files()

    input_fn = input_fn_builder_unmasked(input_files, FLAGS, False, True)
    model_fn = model_fn_share_fetch_grad(config, train_config, SharingFetchGradModel)
    run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
