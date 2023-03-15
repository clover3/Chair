import logging
import sys
import tensorflow as tf
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import \
    get_vector_regression_dataset, get_dummy_vector_regression_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import get_transformer_sparse_encoder, \
    get_regression_model2, get_dummy_regression_model
from trainer_v2.per_project.transparency.splade_regression.trainer_vector_regression import TrainerVectorRegression
from trainer_v2.train_util.arg_flags import flags_parser
from transformers import AutoTokenizer
import numpy as np


def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    c_log.setLevel(logging.DEBUG)

    model_config = {
        "model_type": "distilbert-base-uncased",
    }
    vocab_size = AutoTokenizer.from_pretrained(model_config["model_type"]).vocab_size

    dataset_info = {
        "max_seq_length": 256,
        "max_vector_indices": 512,
        "vocab_size": vocab_size
    }

    def model_factory():
        # new_model = get_regression_model(model_config, run_config.is_training())
        c_log.info("Using dummy regression model")
        new_model = get_dummy_regression_model(model_config, run_config.is_training())
        return new_model

    trainer: TrainerIF = TrainerVectorRegression(
        model_config, run_config, model_factory)

    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig300_3()
    trainer: TrainerIF = Trainer(bert_params, model_config, run_config, task_model)

    def build_dataset(input_files, is_for_training):
        # return get_vector_regression_dataset(
        #     input_files, dataset_info, run_config, is_for_training)
        c_log.info("Using dummy regression data")
        return get_dummy_vector_regression_dataset(
            input_files, dataset_info, run_config, is_for_training
        )

    tf_run(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


