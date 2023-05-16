import logging
import os

import trainer_v2.per_project.transparency.mmp.probe.probe_common
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay

import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
import tensorflow as tf

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, load_stock_weights_encoder_only, \
    load_stock_weights_bert_like, load_bert_checkpoint
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF, TrainerCommon
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run, tf_run_train
from trainer_v2.custom_loop.trainer_if import TrainerIF, TrainerIFBase, EmptyEvalObject
from trainer_v2.per_project.transparency.mmp.tt_model.dataset_factories_bow import get_bow_pairwise_dataset, \
    get_bow_pairwise_dataset_qtw
from trainer_v2.per_project.transparency.mmp.tt_model.tt1 import TranslationTable3
from trainer_v2.per_project.transparency.mmp.tt_model.model_conf_defs import InputShapeConfigTT, InputShapeConfigTT100_4
from trainer_v2.train_util.arg_flags import flags_parser
from typing import List, Iterable, Callable, Dict, Tuple, Set


class TranslationTableWBert(ModelV2IF):
    def __init__(self, input_shape: InputShapeConfigTT):
        self.inner_model = None
        self.model: tf.keras.models.Model = None
        self.input_shape: InputShapeConfigTT = input_shape
        self.metrics = None

    def build_model(self, run_config):
        bert_params = load_bert_config(get_bert_config_path())
        print(bert_params)
        self.inner_model = TranslationTable3(bert_params, self.input_shape)
        metrics = {}
        for name in ["verbosity_loss", "acc_loss"]:
            metrics[name] = tf.keras.metrics.Mean(name=name)
        self.metrics = metrics

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.inner_model.model

    def init_checkpoint(self, init_checkpoint):
        if init_checkpoint is None:
            c_log.info("Checkpoint is not specified. ")
        else:
            for bert_cls in self.inner_model.bert_cls_list:
                load_bert_checkpoint(bert_cls, init_checkpoint)

    def get_train_metrics(self):
        return self.metrics

class Trainer(TrainerIFBase):
    def __init__(self, run_config: RunConfig2,
                 inner_model: TranslationTableWBert):
        self.run_config = run_config
        self.eval_metrics = {}
        self.eval_metrics_factory = {}
        self.batch_size = run_config.common_run_config.batch_size
        self.inner_model = inner_model

        # These variables will be initialized by build_model()
        self.model = None
        self.optimizer = None

    def get_optimizer(self):
        return AdamWeightDecay(
            learning_rate=self.run_config.train_config.learning_rate,
            exclude_from_weight_decay=[]
        )

    def build_model(self):
        if self.run_config.is_training():
            self.inner_model.build_model(self.run_config)
            self.model = self.inner_model.get_keras_model()
            self.model.summary(140)
            self.train_metrics = self.inner_model.get_train_metrics()
            self.optimizer = self.get_optimizer()
            self.model.optimizer = self.optimizer
        else:
            pass
        for k, v in self.eval_metrics_factory.items():
            self.eval_metrics[k] = v()

    def do_init_checkpoint(self, init_checkpoint):
        self.inner_model.init_checkpoint(init_checkpoint)

    def set_keras_model(self, model):
        self.model = model

    def get_keras_model(self):
        return self.model

    @tf.function
    def train_step(self, item):
        model = self.get_keras_model()
        with tf.GradientTape() as tape:
            predictions, losses, verbosity_loss = model(item, training=True)
            loss = tf.reduce_mean(losses)
            self.train_metrics['verbosity_loss'].update_state(verbosity_loss)
            self.train_metrics['acc_loss'].update_state(losses - verbosity_loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        apply_gradient_warning_less(self.optimizer, gradients, model.trainable_variables)
        return loss

    def get_train_metrics(self) -> Dict[str, trainer_v2.per_project.transparency.mmp.probe.probe_common.Metric]:
        ret = self.inner_model.get_train_metrics()
        return ret

    def get_eval_metrics(self) -> Dict[str, trainer_v2.per_project.transparency.mmp.probe.probe_common.Metric]:
        return self.eval_metrics

    def train_callback(self):
        try:
            self.model.callback({'step': self.optimizer.iterations})
        except AttributeError:
            pass

    def get_eval_object(self, eval_batches, strategy):
        eval_object = EmptyEvalObject()
        return eval_object


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    input_shape = InputShapeConfigTT100_4()
    BertUsingModel = TranslationTableWBert(input_shape)
    trainer: TrainerIFBase = Trainer(run_config, BertUsingModel)

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


