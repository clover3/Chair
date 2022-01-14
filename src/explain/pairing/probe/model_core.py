import tensorflow as tf

from explain.pairing.common_unit import probe_modeling
from explain.pairing.match_predictor import ProbeConfigI
from models.transformer.transformer_cls import transformer_pooled_I, transformer_pooled


class ProbeSet:
    def __init__(self,
                 main_model: transformer_pooled_I,
                 hp,
                 per_layer_component: str,
                 use_embedding_out: bool
                 ):
        all_layers = main_model.get_all_encoder_layers()  # List[ tensor[batch,seq_length, hidden] ]
        if use_embedding_out:
            all_layers = [main_model.get_embedding_output()] + all_layers
        num_labels = hp.num_classes
        self.main_model = main_model
        _, input_mask, _ = main_model.get_input_placeholders()
        probe_target = tf.expand_dims(tf.nn.softmax(main_model.get_logits(), axis=-1), axis=1)
        probe_target = tf.tile(probe_target, [1, hp.seq_max, 1])
        if per_layer_component == 'linear':
            def network(layer_output_fixed):
                logits = tf.layers.dense(layer_output_fixed, num_labels)
                return logits
        elif per_layer_component == 'mlp':
            def network(layer_output_fixed):
                hidden = tf.layers.dense(layer_output_fixed, 768, activation='relu')
                logits = tf.layers.dense(hidden, num_labels)
                return logits
        else:
            assert False

        with tf.variable_scope("match_predictor"):
            per_layer_models = list([probe_modeling(layer, probe_target, input_mask, network)
                                     for layer_no, layer in enumerate(all_layers)])
        self.per_layer_models = per_layer_models

        loss = 0
        for d in per_layer_models:
            loss += d.loss

        self.all_losses = tf.stack([d.loss for d in per_layer_models])
        self.loss = loss
        self.per_layer_logits = list([d.logits for d in per_layer_models])


def build_model(hp, lms_model_config: ProbeConfigI, is_training):
    main_model = transformer_pooled(hp, hp.vocab_size, is_training)
    probe_set = ProbeSet(main_model,
                         hp,
                         lms_model_config.per_layer_component,
                         lms_model_config.use_embedding_out
                         )
    return main_model, probe_set