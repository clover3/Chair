import logging
import os

from trainer_v2.custom_loop.dataset_factories import read_pairwise_as_pointwise
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.per_project.transparency.mmp.trainer_d_out2 import TrainerDOut2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.per_project.transparency.mmp.feature_extractor.feature_building import ProbeOnBERT, ProbeLossFromDict

import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
import tensorflow as tf

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF, ModelV3IF
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run2
from trainer_v2.custom_loop.trainer_if import TrainerIF, TrainerIFBase
from trainer_v2.per_project.transparency.mmp.tt_model.model_conf_defs import InputShapeConfigTT, InputShapeConfigTT100_4
from trainer_v2.train_util.arg_flags import flags_parser


class ProbeModel(ModelV3IF):
    def __init__(self, input_shape: InputShapeConfigTT):
        self.inner_model = None
        self.model: tf.keras.models.Model = None
        self.loss = None
        self.input_shape: InputShapeConfigTT = input_shape
        self.log_var = ["loss"]

        for i in [0, 1, 12]:
            self.log_var.append("layer_{}_loss".format(i))

    def build_model(self, run_config):
        init_checkpoint = run_config.train_config.init_checkpoint
        c_log.info("Loading model from {}".format(init_checkpoint))
        ranking_model = tf.keras.models.load_model(init_checkpoint, compile=False)
        self.inner_model = ProbeOnBERT(ranking_model)
        self.loss = ProbeLossFromDict()

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.inner_model.model

    def init_checkpoint(self, init_checkpoint):
        pass

    def get_train_metrics(self):
        return {}

    def get_train_metrics_for_summary(self):
        return self.inner_model.get_metrics()

    def get_loss_fn(self):
        return self.loss


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    input_shape = InputShapeConfigTT100_4()
    model_v2 = ProbeModel(input_shape)
    trainer: TrainerIFBase = TrainerDOut2(run_config, model_v2)

    def build_dataset(input_files, is_for_training):
        return read_pairwise_as_pointwise(
            input_files, run_config, ModelConfig256_1(), is_for_training)

    tf_run2(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


