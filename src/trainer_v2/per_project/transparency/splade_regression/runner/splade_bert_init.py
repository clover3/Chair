import logging
import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
import tensorflow as tf

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, load_stock_weights_encoder_only, \
    load_stock_weights_bert_like
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import \
    get_vector_regression_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.bert_vector_regression import BertVectorRegression
from trainer_v2.per_project.transparency.splade_regression.trainer_vector_regression import TrainerVectorRegression
from trainer_v2.custom_loop.prediction_trainer import ModelV3IF
from trainer_v2.train_util.arg_flags import flags_parser
from transformers import AutoTokenizer
import numpy as np


class BERT_VR(ModelV3IF):
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info
        self.inner_model = None
        self.model: tf.keras.models.Model = None

    def build_model(self, run_config):
        bert_params = load_bert_config(get_bert_config_path())
        self.inner_model = BertVectorRegression(bert_params, self.dataset_info)

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.inner_model.model

    def init_checkpoint(self, init_checkpoint):
        if init_checkpoint is None:
            c_log.info("Checkpoint is not specified. ")
        else:
            load_stock_weights_bert_like(self.inner_model.l_bert, init_checkpoint, n_expected_restore=197)


def main(args):
    c_log.info(__file__)

    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    vocab_size = 30522

    dataset_info = {
        "max_seq_length": 256,
        "max_vector_indices": 512,
        "vocab_size": vocab_size
    }

    trainer: TrainerIF = TrainerVectorRegression(run_config, BERT_VR(dataset_info))

    def build_dataset(input_files, is_for_training):
        return get_vector_regression_dataset(
            input_files, dataset_info, run_config, is_for_training)

    tf_run(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


