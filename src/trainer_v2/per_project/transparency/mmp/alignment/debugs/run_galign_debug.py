import logging
import os
import code

from trainer_v2.custom_loop.dataset_factories import read_pairwise_as_pointwise
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.per_project.transparency.mmp.alignment.dataset_factory import read_galign
from trainer_v2.per_project.transparency.mmp.trainer_d_out2 import TrainerDOut2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.per_project.transparency.mmp.probe.probe_network import ProbeOnBERT, ProbeLossFromDict
from trainer_v2.per_project.transparency.mmp.probe.align_network import AlignLossFromDict, GAlignNetwork

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


class GAlignModel(ModelV3IF):
    def __init__(self, input_shape: InputShapeConfigTT):
        self.network = None
        self.model: tf.keras.models.Model = None
        self.loss = None
        self.input_shape: InputShapeConfigTT = input_shape

    def build_model(self, run_config):
        init_checkpoint = run_config.train_config.init_checkpoint
        c_log.info("Loading model from {}".format(init_checkpoint))
        ranking_model = tf.keras.models.load_model(init_checkpoint, compile=False)
        self.network = GAlignNetwork(ranking_model)
        self.loss = AlignLossFromDict()

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        pass

    def get_train_metrics(self):
        return {}

    def get_train_metrics_for_summary(self):
        return self.network.get_align_metrics()

    def get_loss_fn(self):
        return self.loss





if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    input_shape = InputShapeConfigTT100_4()
    model_v3 = GAlignModel(input_shape)
    trainer: TrainerIFBase = TrainerDOut2(run_config, model_v3)

    def build_dataset(input_files, is_for_training):
        return read_galign(
            input_files, run_config, ModelConfig256_1(), is_for_training)

    dataset_factory = build_dataset
    train_dataset = dataset_factory(run_config.dataset_config.train_files_path, True)
    trainer.build_model()
    trainer.do_init_checkpoint(run_config.train_config.init_checkpoint)

    model = trainer.get_keras_model()
    train_itr = iter(train_dataset)
    batch_item = next(train_itr)
    output = model(batch_item)
    # TODO check why attn_probs is zero
    #

    print(output["align_feature_d"])
    code.interact(local=locals())
