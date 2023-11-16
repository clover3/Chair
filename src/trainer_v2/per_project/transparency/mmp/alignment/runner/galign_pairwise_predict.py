import os
import pickle
from dataclasses import dataclass


from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.mmp.alignment.dataset_factory import read_galign, read_galign_v2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
import tensorflow as tf
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2, get_run_config_for_predict
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config_for_predict(args)
    run_config.print_info()

    def build_dataset(input_files, is_for_training):
        return read_galign_v2(
            input_files, run_config, is_for_training)
    strategy = get_strategy_from_config(run_config)

    with strategy.scope():
        eval_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
        model = tf.keras.models.load_model(run_config.predict_config.model_save_path, compile=False)
        outputs = model.predict(eval_dataset)
        pickle.dump(outputs, open(run_config.predict_config.predict_save_path, "wb"))


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


