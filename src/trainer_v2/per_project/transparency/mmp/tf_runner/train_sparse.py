import logging
import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
import tensorflow as tf

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, load_stock_weights_encoder_only, \
    load_stock_weights_bert_like
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run, tf_run_train
from trainer_v2.custom_loop.train_loop_helper import eval_tensor
from trainer_v2.custom_loop.trainer_if import TrainerIF, TrainerIFBase
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import \
    get_vector_regression_dataset, get_three_text_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.bert_sparse_encoder import BertSparseEncoder, \
    PairwiseTrainBertSparseEncoder
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel


class RegWeightScheduler:
    def __init__(self, lambda_, T):
        self.lambda_ = lambda_
        self.T = T

    def get_lambda(self, cur_step):
        loc = min(cur_step, self.T)
        lambda_t = self.lambda_ * (loc / self.T) ** 2
        return lambda_t


class ModelWrap(ModelV2IF):
    def __init__(self, dataset_info, reg_lambda, warmup_step):
        self.dataset_info = dataset_info
        self.inner_model = None
        self.model: tf.keras.models.Model = None
        self.reg_schedule = RegWeightScheduler(reg_lambda, warmup_step)
        self.last_alpha = -1

    def build_model(self, run_config):
        bert_params = load_bert_config(get_bert_config_path())
        self.inner_model = PairwiseTrainBertSparseEncoder(bert_params, self.dataset_info)

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.inner_model.model

    def init_checkpoint(self, init_checkpoint):
        if init_checkpoint is None:
            c_log.info("Checkpoint is not specified. ")
        else:
            load_stock_weights_bert_like(self.inner_model.l_bert, init_checkpoint, n_expected_restore=197)

    def callback(self, info):
        step = info['step']
        step = eval_tensor(step)
        alpha = self.reg_schedule.get_lambda(step)

        if self.last_alpha != alpha:
            c_log.info("Step %d update alpha to %f", step, alpha)
            alpha_tensor = self.inner_model.reg_fn.alpha
            alpha_tensor.assign(tf.cast(alpha, tf.float32))
            self.last_alpha = alpha



@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    vocab_size = 30522
    dataset_info = {
        "max_seq_length": 256,
        "vocab_size": vocab_size
    }
    reg_lambda = 0.5
    reg_warmup_steps = 50000
    model = ModelWrap(dataset_info, reg_lambda, reg_warmup_steps)
    trainer: TrainerIFBase = TrainerForLossReturningModel(run_config, model)

    def build_dataset(input_files, is_for_training):
        return get_three_text_dataset(
            input_files, dataset_info, run_config, is_for_training, True)

    if run_config.is_training():
        tf_run_train(run_config, trainer, build_dataset)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

