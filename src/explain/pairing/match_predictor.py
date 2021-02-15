from abc import ABC
from typing import NamedTuple

import tensorflow as tf

from explain.explain_model import ExplainModeling
from models.transformer.transformer_cls import transformer_pooled

Tensor = tf.Tensor


class MatchPredictor:
    def __init__(self,
                 main_model: transformer_pooled,
                 ex_model: ExplainModeling,
                 target_ex_idx: int,
                 per_layer_component: str,
                 use_embedding_out: bool
                 ):
        all_layers = main_model.model.get_all_encoder_layers()  # List[ tensor[batch,seq_length, hidden] ]
        if use_embedding_out:
            all_layers = [main_model.model.get_embedding_output()] + all_layers

        self.main_model = main_model
        _, input_mask, _ = main_model.x_list
        self.ex_model = ex_model
        ex_scores = ex_model.get_ex_scores(target_ex_idx)  # tensor[batch, seq_length]
        if per_layer_component == 'linear':
            network = linear
        elif per_layer_component == 'mlp':
            network = mlp
        else:
            assert False

        with tf.variable_scope("match_predictor"):
            per_layer_models = list([per_layer_modeling(layer, ex_scores, input_mask, network)
                                     for layer_no, layer in enumerate(all_layers)])
        self.per_layer_models = per_layer_models

        loss = 0
        for d in per_layer_models:
            loss += d.loss

        self.all_losses = tf.stack([d.loss for d in per_layer_models])
        self.loss = loss
        self.per_layer_logits = list([d.logits for d in per_layer_models])



class PerLayerModel(NamedTuple):
    loss: Tensor  # []
    losses: Tensor
    logits: Tensor  # [batch, seq_length, 2]
    error: Tensor  # [batch, seq_length]


def per_layer_modeling(layer_output, ex_scores_raw, input_mask, network):
    ex_scores = tf.stop_gradient(ex_scores_raw)
    layer_output_fixed = tf.stop_gradient(layer_output)
    logits = network(layer_output_fixed)
    ex_score0 = 1-ex_scores
    true_y = tf.stack([ex_score0, ex_scores], axis=2)
    per_token_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=true_y)
    per_token_losses = tf.reduce_sum(per_token_losses, axis=2)

    input_mask_float = tf.cast(input_mask, tf.float32)
    per_token_losses = per_token_losses * input_mask_float # [batch, seq_length]
    num_valid = tf.reduce_sum(input_mask_float, axis=1) + 1e-6
    pred_prob = tf.nn.softmax(logits, axis=-1)[:, :, 1]
    error = tf.abs(pred_prob - ex_scores)
    losses = tf.reduce_sum(per_token_losses, axis=1) / num_valid
    loss = tf.reduce_mean(losses)
    return PerLayerModel(loss, losses, logits, error)


def linear(layer_output_fixed):
    logits = tf.layers.dense(layer_output_fixed, 2)
    return logits


def mlp(layer_output_fixed):
    hidden = tf.layers.dense(layer_output_fixed, 768, activation='tanh')
    logits = tf.layers.dense(hidden, 2)
    return logits


class LMSConfigI(ABC):
    num_tags = None
    target_idx = None
    use_embedding_out = None
    per_layer_component = None


def build_model(ex_modeling_class, hparam, lms_model_config: LMSConfigI):
    main_model = transformer_pooled(hparam, hparam.vocab_size)
    ex_model = ex_modeling_class(main_model.model.sequence_output,
                                 hparam.seq_max,
                                 lms_model_config.num_tags,
                                 main_model.batch2feed_dict)

    match_predictor = MatchPredictor(main_model,
                                     ex_model,
                                     lms_model_config.target_idx,
                                     lms_model_config.per_layer_component,
                                     lms_model_config.use_embedding_out
                                     )
    return main_model, ex_model, match_predictor


class LMSConfig(LMSConfigI):
    num_tags = 3
    target_idx = 2
    use_embedding_out = False
    per_layer_component = 'linear'


class LMSConfig2(LMSConfigI):
    num_tags = 3
    target_idx = 1
    use_embedding_out = True
    per_layer_component = 'linear'
