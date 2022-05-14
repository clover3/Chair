import tensorflow as tf
from official.modeling import performance
from official.nlp.modeling import networks

from trainer_v2.partial_processing.config_helper import MultiSegModelConfig
from trainer_v2.partial_processing.dev_modeling import get_transformer_encoder, get_optimizer
from trainer_v2.run_config import RunConfigEx


def vector_three_feature(v1, v2):
    concat_layer = tf.keras.layers.Concatenate()
    concat = concat_layer([v1, v2])
    sub = v1 - v2
    dot = tf.multiply(v1, v2)
    output = tf.concat([sub, dot, concat], axis=-1)
    return output


def model_fn_factory(model_config: MultiSegModelConfig, run_config: RunConfigEx):
    bert_config = model_config.bert_config
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=bert_config.initializer_range)

    def model_fn():
        bert_encoder1: tf.keras.Model = get_transformer_encoder(
            bert_config, input_prefix_fix="encoder1_")
        bert_encoder2: tf.keras.Model = get_transformer_encoder(
            bert_config, input_prefix_fix="encoder2_")

        def get_from_bert_encoder(bert_encoder):
            inputs = bert_encoder.inputs
            outputs = bert_encoder(inputs)
            cls_output = outputs['pooled_output']
            return cls_output

        cls_output1 = get_from_bert_encoder(bert_encoder1)
        cls_output2 = get_from_bert_encoder(bert_encoder2)

        # concat_layer = tf.keras.layers.Concatenate()
        # cls_concat = concat_layer([cls_output1, cls_output2])
        feature_rep = vector_three_feature(cls_output1, cls_output2)
        classifier = networks.Classification(
            input_width=feature_rep.shape[-1],
            num_classes=model_config.num_classes,
            initializer=initializer,
            output='logits',
            name='sentence_prediction')
        predictions = classifier(feature_rep)
        inputs = [bert_encoder1.inputs, bert_encoder2.inputs]

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