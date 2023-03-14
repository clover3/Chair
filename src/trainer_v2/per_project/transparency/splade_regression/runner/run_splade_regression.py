import logging
import sys
import tensorflow as tf
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
from trainer_v2.custom_loop.per_task.trainer_vector_regression import TrainerVectorRegression
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import \
    get_vector_regression_dataset, get_dummy_vector_regression_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import get_regression_model, \
    get_regression_model2, get_dummy_regression_model
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF
from trainer_v2.train_util.arg_flags import flags_parser
from transformers import AutoTokenizer
import numpy as np


class DistilBertVR(ModelV2IF):
    def __init__(self, model_config):
        self.model_config = model_config
        self.model: tf.keras.models.Model = None
        pass

    def build_model(self, run_config):
        self.model = get_regression_model(self.model_config, run_config.is_training())

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.model

    def init_checkpoint(self, model_path):
        pass

def ignore_distilbert_save_warning():
    ignore_msg = [
        r"Skipping full serialization of Keras layer .*Dropout",
        r"Found untraced functions such as"
    ]
    ignore_filter = IgnoreFilterRE(ignore_msg)
    tf_logging = logging.getLogger("tensorflow")
    tf_logging.addFilter(ignore_filter)


def main(args):
    c_log.info(__file__)
    ignore_distilbert_save_warning()

    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    model_config = {
        "model_type": "distilbert-base-uncased",
    }
    vocab_size = AutoTokenizer.from_pretrained(model_config["model_type"]).vocab_size

    dataset_info = {
        "max_seq_length": 256,
        "max_vector_indices": 512,
        "vocab_size": vocab_size
    }

    trainer: TrainerIF = TrainerVectorRegression(run_config, DistilBertVR(model_config))

    def build_dataset(input_files, is_for_training):
        return get_vector_regression_dataset(
            input_files, dataset_info, run_config, is_for_training)

    tf_run(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


