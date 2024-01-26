import tensorflow as tf

from trainer_v2.custom_loop.definitions import HFModelConfigType
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.neural_network_def.two_seg_two_model import TwoSegConcatLogitCombineTwoModel
from trainer_v2.custom_loop.per_task.ts_util import get_local_decision_layer_from_model_by_shape


def load_ts_concat_local_decision_model(
        new_model_config: HFModelConfigType, model_save_path):
    task_model = TwoSegConcatLogitCombineTwoModel(
        new_model_config, CombineByScoreAdd)
    task_model.build_model(None)
    model = task_model.point_model
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(model_save_path).expect_partial()
    model.summary()
    local_decision_layer = get_local_decision_layer_from_model_by_shape(
        model, new_model_config.num_classes)
    new_outputs = [local_decision_layer.output, model.outputs]
    new_model = tf.keras.models.Model(inputs=model.input, outputs=new_outputs)
    return new_model