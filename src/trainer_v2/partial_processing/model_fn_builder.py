import tensorflow as tf
from official.modeling import performance

from trainer_v2.chair_logging import c_log
from trainer_v2.partial_processing.config_helper import MultiSegModelConfig
from trainer_v2.partial_processing.modeling import get_transformer_encoder, get_optimizer
from trainer_v2.run_config import RunConfigEx


def model_factory_cls(model_config: MultiSegModelConfig, run_config: RunConfigEx):
    bert_config = model_config.bert_config

    def model_fn():
        c_log.debug("model_fn() 1")
        bert_encoder: tf.keras.Model = get_transformer_encoder(
            bert_config)
        c_log.debug("model_fn() 2")
        inputs = bert_encoder.inputs
        outputs = bert_encoder(inputs)
        c_log.debug("model_fn() 3")
        cls_output = outputs['pooled_output']
        predictions = tf.keras.layers.Dense(model_config.num_classes, name="sentence_prediction")(cls_output)
        outer_model = tf.keras.Model(
            inputs=bert_encoder.inputs,
            outputs=predictions)
        c_log.debug("model_fn() 4")
        optimizer = get_optimizer(run_config)
        c_log.debug("model_fn() 4.5")
        outer_model.optimizer = performance.configure_optimizer(optimizer)
        inner_model_list = [bert_encoder]
        c_log.debug("model_fn() 5")
        return outer_model, inner_model_list
    return model_fn