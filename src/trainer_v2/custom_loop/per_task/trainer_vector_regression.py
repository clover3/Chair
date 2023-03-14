import tensorflow as tf

from trainer_v2.custom_loop.eval_util import EvalObject, MultipleEvalObject
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF, PredictionTrainerCommon
from trainer_v2.custom_loop.run_config2 import RunConfig2


@tf.function
def squared_error(pred, label):
    diff = pred-label
    return tf.square(diff)


class TrainerVectorRegression(PredictionTrainerCommon):
    def __init__(self, run_config: RunConfig2,
                 inner_model: ModelV2IF):
        super(TrainerVectorRegression, self).__init__(run_config, inner_model)
        print("TrainerVectorRegression")
        self.loss_fn_inner = squared_error

    def get_eval_object(self, eval_batches, strategy):
        print("Pass self.loss_fn to eval object")
        loss_eval_obj = EvalObject(self.model, eval_batches, strategy, self.loss_fn, {})
        other_eval_objects = []

        return MultipleEvalObject(loss_eval_obj, other_eval_objects)

    def loss_fn(self, labels, predictions):
        per_dimension_loss = self.loss_fn_inner(labels, predictions)
        per_example_loss = tf.reduce_sum(per_dimension_loss, axis=1)
        avg_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size)
        return avg_loss

    def get_optimizer(self):
        return AdamWeightDecay(
            learning_rate=self.run_config.train_config.learning_rate,
            exclude_from_weight_decay=[]
        )