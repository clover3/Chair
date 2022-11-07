from typing import Callable, Dict
import tensorflow as tf

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.reinforce.monte_carlo_policy_function import PolicyFunction
from trainer_v2.evidence_selector.seq_pred_policy_gradient import SeqPredREINFORCE


class PolicyGradientTrainer(TrainerIF):
    def __init__(self, bert_params: object, model_config: object,
                 run_config: RunConfig2,
                 inner_model: ClassificationModelIF,
                 policy_func_factory: Callable[[tf.keras.models.Model], PolicyFunction],
                 reinforce: SeqPredREINFORCE,
                 ):
        self.bert_params = bert_params
        self.model_config = model_config
        self.run_config = run_config
        self.eval_metrics = {}
        self.eval_metrics_factory = {
            'ref_accuracy': lambda: tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
        }
        self.batch_size = run_config.common_run_config.batch_size
        self.inner_model = inner_model

        # These variables will be initialized by build_model()
        self.train_metrics = None
        self.keras_model = None
        self.optimizer = None
        self.loss_fn_inner = None
        self.policy_func_factory = policy_func_factory
        self.reinforce = reinforce
        self.policy_func: PolicyFunction = None

    def build_model(self):
        c_log.info("build model")
        run_config = self.run_config
        if self.run_config.is_training():
            self.inner_model.build_model(self.bert_params, self.model_config)
            self.keras_model = self.inner_model.get_keras_model()
            self.policy_func = self.policy_func_factory(self.keras_model)
            self.reinforce.init(self.policy_func)
            self.keras_model.summary(140)
            optimizer = AdamWeightDecay(learning_rate=run_config.train_config.learning_rate)
            self.keras_model.optimizer = optimizer
            self.train_metrics = {}
            self.optimizer = optimizer
        else:
            pass
        for k, v in self.eval_metrics_factory.items():
            self.eval_metrics[k] = v()
        self.loss_fn_inner = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    def set_keras_model(self, model):
        self.keras_model = model

    def loss_fn(self, labels, predictions):
        per_example_loss = self.loss_fn_inner(labels, predictions)
        loss =  tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)
        return loss

    def get_keras_model(self):
        return self.keras_model

    def do_init_checkpoint(self, init_checkpoint):
        return self.inner_model.init_checkpoint(init_checkpoint)

    def train_step(self, item):
        model = self.get_keras_model()
        state = item['state']
        sample_actions = item['sample_actions']
        base_reward = item['base_reward']
        sample_reward_list = item['sample_reward_list']

        with tf.GradientTape() as tape:
            log_p_action = self.policy_func.get_log_action_prob(state, sample_actions)
            added_reward = sample_reward_list - tf.expand_dims(base_reward, axis=1)
            loss = - tf.multiply(added_reward, log_p_action)
            loss = tf.reduce_sum(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        apply_gradient_warning_less(self.optimizer, gradients, model.trainable_variables)
        return loss

    def get_train_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.train_metrics

    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.eval_metrics

    def train_callback(self):
        try:
            self.inner_model.callback({'step': self.optimizer.iterations})
        except AttributeError:
            pass

    def build_dataset(self, src_path, is_training) -> tf.data.Dataset:
        return self.reinforce.get_dataset(src_path, is_training)


