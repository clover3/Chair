import os

from transformers import AutoTokenizer

from trainer_v2.per_project.tli.model_load_h5 import load_weights_from_hdf5
from trainer_v2.per_project.transparency.mmp.alignment.dataset_factory import read_galign_pairwise
from trainer_v2.per_project.transparency.mmp.alignment.network.first_hidden_pairwise import GAlignFirstHiddenPairwise
from trainer_v2.per_project.transparency.mmp.alignment.network.common import AlignLossFromDictPairConcat
from trainer_v2.per_project.transparency.mmp.alignment.runner.run_pairwise_common import run_galign_pairwise_training
from trainer_v2.per_project.transparency.mmp.probe.align_network import AddLosses
from trainer_v2.per_project.transparency.mmp.trainer_d_out2 import TrainerDOut2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import sys
from trainer_v2.chair_logging import c_log
import tensorflow as tf
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.prediction_trainer import ModelV3IF
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run2
from trainer_v2.custom_loop.trainer_if import TrainerIFBase
from trainer_v2.train_util.arg_flags import flags_parser


class GAlignModel(ModelV3IF):
    def __init__(self):
        self.network = None
        self.model: tf.keras.models.Model = None
        self.loss = None

    def build_model(self, run_config):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.network = GAlignFirstHiddenPairwise(tokenizer)
        loss_list = [
            AlignLossFromDictPairConcat(),
        ]
        self.loss = AddLosses(loss_list)

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        def name_mapping(name, prefix):
            return f"tf_bert_for_sequence_classification/{name}"

        c_log.info("Loading model from {}".format(init_checkpoint))
        n_param_emb = 5
        n_param_per_layer = 16
        n_param = n_param_emb + n_param_per_layer
        load_weights_from_hdf5(self.network.model, init_checkpoint, name_mapping, n_param)

    def get_train_metrics(self):
        return {}

    def get_train_metrics_for_summary(self):
        metrics = self.network.get_align_metrics()
        return metrics

    def get_eval_metrics_for_summary(self):
        metrics = self.network.get_align_metrics()
        return metrics

    def get_loss_fn(self):
        return self.loss


@report_run3
def main(args):
    c_log.info(__file__)
    model_v3 = GAlignModel()
    run_galign_pairwise_training(args, model_v3)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
