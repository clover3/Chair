import logging
import sys

import tensorflow as tf
from transformers import AutoTokenizer

from misc_lib import path_join
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import \
    get_vector_regression_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import get_regression_model
from trainer_v2.train_util.arg_flags import flags_parser

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import get_run_config2, RunConfig2


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    strategy = get_strategy_from_config(run_config)

    model_config = {
        "model_type": "distilbert-base-uncased",
    }
    vocab_size = AutoTokenizer.from_pretrained(model_config["model_type"]).vocab_size
    dataset_info = {
        "max_seq_length": 256,
        "max_vector_indices": 512,
        "vocab_size": vocab_size
    }

    def build_dataset():
        input_files = run_config.dataset_config.train_files_path
        return get_vector_regression_dataset(input_files, dataset_info, run_config, True)

    with strategy.scope():
        new_model = get_regression_model(model_config, True)
        new_model.compile(loss="MSE", optimizer="adam")
        dataset = build_dataset()
        train_steps = 10000
        new_model.fit(dataset, epochs=1, steps_per_epoch=train_steps)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
