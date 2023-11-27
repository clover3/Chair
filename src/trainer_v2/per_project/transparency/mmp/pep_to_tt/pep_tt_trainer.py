import tensorflow as tf
from trainer_v2.custom_loop.prediction_trainer import TrainerCommon, ModelV2IF
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import apply_gradient_warning_less


class PEP_TT_Trainer(TrainerCommon):
    def __init__(self, run_config: RunConfig2,
                 inner_model: ModelV2IF,
                 eval_object_factory=None
                 ):
        super(PEP_TT_Trainer, self).__init__(run_config, inner_model)
        self.eval_object_factory = eval_object_factory

    def get_optimizer(self):
        return AdamWeightDecay(
            learning_rate=self.get_learning_rate(),
            exclude_from_weight_decay=[]
        )

    def train_step(self, item):
        model = self.get_keras_model()
        with tf.GradientTape() as tape:
            predictions, loss = model(item, training=True)

        gradients = tape.gradient(loss, model.trainable_variables)
        apply_gradient_warning_less(self.optimizer, gradients, model.trainable_variables)
        return loss
