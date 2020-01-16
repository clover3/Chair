
import tensorflow as tf

from taskman_client.wrapper import report_run
from tlm.model_cnfig import JsonConfig
from tlm.sero.sero_model_fn import model_fn_sero_ranking_predict
from tlm.training.flags_wrapper import get_input_files_from_flags
from tlm.training.input_fn import input_fn_builder_prediction
from tlm.training.train_config import LMTrainConfig
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = LMTrainConfig.from_flags(FLAGS)
    input_file = get_input_files_from_flags(FLAGS)
    input_fn = input_fn_builder_prediction(input_file, config.total_sequence_length)
    model_fn = model_fn_sero_ranking_predict(config, train_config, FLAGS.modeling)
    return run_estimator(model_fn, input_fn)

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
