from trainer_v2.chair_logging import c_log
import os
from trainer_v2.custom_loop.definitions import ModelConfig256_1, ModelConfig300_3
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.per_project.tli.tli_probe.runner.dev_init_model import load_weights_from_hdf5
from trainer_v2.per_project.tli.tli_probe.tli_dataset import get_tli_dataset
from trainer_v2.per_project.tli.tli_probe.tli_probe_network import TliProbe, ProbeBCE
from trainer_v2.per_project.transparency.mmp.trainer_d_out2 import TrainerDOut2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf

from cpath import get_bert_config_path
from trainer_v2.custom_loop.prediction_trainer import ModelV3IF
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run2
from trainer_v2.custom_loop.trainer_if import TrainerIF, TrainerIFBase
from trainer_v2.per_project.transparency.mmp.tt_model.model_conf_defs import InputShapeConfigTT, InputShapeConfigTT100_4
from trainer_v2.train_util.arg_flags import flags_parser


class TliProbeModel(ModelV3IF):
    def __init__(self, input_shape: InputShapeConfigTT):
        self.network = None
        self.model: tf.keras.models.Model = None
        self.loss = None
        self.input_shape: InputShapeConfigTT = input_shape

    def build_model(self, run_config):
        init_checkpoint = run_config.train_config.init_checkpoint
        c_log.info("Loading model from {}".format(init_checkpoint))
        bert_params = load_bert_config(get_bert_config_path())
        num_layer = bert_params.num_layers
        bert_params.out_layer_ndxs = list(range(num_layer))
        model_config = ModelConfig300_3()

        self.network = TliProbe(model_config, bert_params)
        self.loss = ProbeBCE()

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        def name_mapping(name, prefix):
            return name

        model = self.network.model
        load_weights_from_hdf5(model, init_checkpoint, name_mapping, 197 + 4)


    def get_train_metrics(self):
        return {}

    def get_train_metrics_for_summary(self):
        return self.network.get_probe_metrics()

    def get_loss_fn(self):
        return self.loss


# @report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    input_shape = InputShapeConfigTT100_4()
    model_v3 = TliProbeModel(input_shape)
    trainer: TrainerIFBase = TrainerDOut2(run_config, model_v3)

    def build_dataset(input_files, is_for_training):
        return get_tli_dataset(
            input_files, run_config, ModelConfig300_3(), is_for_training)

    tf_run2(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


