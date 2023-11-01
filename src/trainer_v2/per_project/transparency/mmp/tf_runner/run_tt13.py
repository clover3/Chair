import logging
import os
from typing import Optional

from trainer_v2.per_project.transparency.mmp.trainer_d_out import TrainerDOut
from trainer_v2.per_project.transparency.mmp.tt_model.net_common import pairwise_hinge
from trainer_v2.per_project.transparency.mmp.tt_model.tt1 import ScoringLayer2, ScoringLayerSigmoidCap, ScoringLayer4, \
    ScoringLayer5
from trainer_v2.per_project.transparency.mmp.tt_model.tt_vectored import TTVectorBasedTrainNetwork

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
import tensorflow as tf

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, load_stock_weights_encoder_only, \
    load_stock_weights_bert_like, load_bert_checkpoint
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run, tf_run_train
from trainer_v2.custom_loop.trainer_if import TrainerIF, TrainerIFBase
from trainer_v2.per_project.transparency.mmp.tt_model.dataset_factories_bow import get_bow_pairwise_dataset, \
    get_bow_pairwise_dataset_qtw
from trainer_v2.per_project.transparency.mmp.tt_model.model_conf_defs import InputShapeConfigTT, InputShapeConfigTT100_4
from trainer_v2.train_util.arg_flags import flags_parser


class TranslationTableVector(ModelV2IF):
    def __init__(self, input_shape: InputShapeConfigTT):
        self.inner_model: Optional[TTVectorBasedTrainNetwork] = None
        self.model: tf.keras.models.Model = None
        self.input_shape: InputShapeConfigTT = input_shape
        self.train_metrics = None
        self.log_var = ["verbosity_loss", 'pairwise_loss', "loss"]

    def build_model(self, run_config):
        bert_params = load_bert_config(get_bert_config_path())
        def scoring_layer_factory():
            return ScoringLayer5(bert_params.hidden_size)

        self.inner_model = TTVectorBasedTrainNetwork(
            bert_params,
            self.input_shape,
            pairwise_hinge,
            scoring_layer_factory
        )

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.inner_model.model

    def init_checkpoint(self, init_checkpoint):
        if init_checkpoint is None:
            c_log.info("Checkpoint is not specified. ")
        else:
            self.inner_model.init_embeddings(init_checkpoint)


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    input_shape = InputShapeConfigTT100_4()
    bert_using_model = TranslationTableVector(input_shape)
    trainer: TrainerIFBase = TrainerDOut(run_config, bert_using_model)
    def build_dataset(input_files, is_for_training):
        return get_bow_pairwise_dataset_qtw(
            input_files, input_shape, run_config, is_for_training)

    run_name = str(run_config.common_run_config.run_name)
    c_log.info("Run name: %s", run_name)

    if run_config.common_run_config.is_debug_run:
        c_log.setLevel(logging.DEBUG)

    if run_config.is_training():
        tf_run_train(run_config, trainer, build_dataset)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


