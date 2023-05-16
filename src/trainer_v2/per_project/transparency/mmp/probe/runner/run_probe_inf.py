import logging
import os

from transformers import AutoTokenizer

from cache import save_to_pickle
from trainer_v2.custom_loop.dataset_factories import get_pairwise_dataset
from trainer_v2.custom_loop.definitions import ModelConfig256_1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
import tensorflow as tf

from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2, get_run_config_for_predict_empty, \
    get_run_config_for_predict
from trainer_v2.custom_loop.train_loop import tf_run2, load_model_by_dir_or_abs
from trainer_v2.train_util.arg_flags import flags_parser


def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config_for_predict(args)
    run_config.print_info()

    def build_dataset(input_files, is_for_training):
        return get_pairwise_dataset(
            input_files, run_config, ModelConfig256_1(), is_for_training, add_dummy_y=False)

    dataset = build_dataset(run_config.dataset_config.eval_files_path, False)

    dataset = dataset.take(1)
    model = load_model_by_dir_or_abs(run_config.predict_config.model_save_path)
    output = model.predict(dataset)

    for b in dataset:
        for i in range(16):
            print(b['input_ids1'][i].numpy().tolist())
            print(b['input_ids2'][i].numpy().tolist())
            print(output['logits'][i].tolist())


    # save_to_pickle(dataset, "probe_dataset")
    # save_to_pickle(output, "probe_inf_dev")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)



