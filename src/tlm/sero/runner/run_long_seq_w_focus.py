

import tensorflow as tf

from taskman_client.wrapper import report_run
from tlm.model.FocusMask import FocusMask
from tlm.model_cnfig import JsonConfig
from tlm.sero.sero_model_fn import model_fn_pooling_long_things
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.input_fn import input_fn_builder_classification_w_focus_mask_data_id
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_file = get_input_files_from_flags(FLAGS)

    input_fn = input_fn_builder_classification_w_focus_mask_data_id(input_file,
                                                        FLAGS,
                                                        FLAGS.do_train)

    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")
    model_fn = model_fn_pooling_long_things(config, train_config, FocusMask, special_flags)
    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
