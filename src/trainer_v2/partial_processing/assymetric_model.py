import tensorflow as tf
from official.modeling import performance

from trainer_v2.partial_processing.bert_encoder_module import get_bert_encoder_module
from trainer_v2.partial_processing.config_helper import MultiSegModelConfig
from trainer_v2.partial_processing.modeling import get_optimizer
from trainer_v2.partial_processing.network_utils import vector_three_feature
from trainer_v2.run_config import RunConfigEx


def add_prefix_to_names(target_module, prefix):
    for module in target_module.submodules:
        layer = module
        layer._name = prefix + layer.name


def model_factory_assym(model_config: MultiSegModelConfig, run_config: RunConfigEx):
    bert_config = model_config.bert_config

    def model_fn():

        def build_keras_input(name):
            return tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=name)
        word_ids1 = build_keras_input('input_ids1')
        mask1 = build_keras_input('input_mask1')
        type_ids1 = build_keras_input('segment_ids1')
        inputs1 = word_ids1, mask1, type_ids1

        word_ids2 = build_keras_input('input_ids2')
        mask2 = build_keras_input('input_mask2')
        type_ids2 = build_keras_input('segment_ids2')
        inputs2 = word_ids2, mask2, type_ids2

        bert_encoder1 = get_bert_encoder_module(bert_config)
        bert_encoder2 = get_bert_encoder_module(bert_config)

        add_prefix_to_names(bert_encoder1, "encoder1/")
        add_prefix_to_names(bert_encoder2, "encoder2/")

        cls_output1 = bert_encoder1(inputs1)
        cls_output2 = bert_encoder2(inputs2)

        feature_rep = vector_three_feature(cls_output1, cls_output2)
        hidden = tf.keras.layers.Dense(model_config.bert_config.hidden_size, activation='relu')(feature_rep)
        predictions = tf.keras.layers.Dense(model_config.num_classes, name="sentence_prediction")(hidden)

        inputs = [inputs1, inputs2]

        outer_model = tf.keras.Model(
            inputs=inputs,
            outputs=predictions)
        optimizer = get_optimizer(run_config)
        outer_model.optimizer = performance.configure_optimizer(optimizer)
        inner_model_list = [bert_encoder1, bert_encoder2]
        return outer_model, inner_model_list

    return model_fn


def fuzzy_logic(logits):
    IDX_CONTRADICTION = 2
    IDX_NEUTRAL = 1
    # logits : [batch_size, max_segment, 3]
    probs = tf.nn.softmax(logits, axis=-1)

    max_probs = tf.reduce_max(probs, axis=1)
    p_c = max_probs[:, IDX_CONTRADICTION]
    p_n = tf.multiply(max_probs[:, IDX_NEUTRAL], (1-p_c))
    p_e = tf.reduce_prod(probs[:, :, 0])

    p_cne_sum = p_e + p_n + p_c

    p_sent_c = tf.divide(p_c, p_cne_sum)
    p_sent_e = tf.divide(p_e, p_cne_sum)
    p_sent_n = tf.divide(p_n, p_cne_sum)
    return tf.stack([p_sent_e, p_sent_n, p_sent_c], axis=-1)
