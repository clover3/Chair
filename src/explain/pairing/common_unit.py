from typing import NamedTuple

import tensorflow as tf

Tensor = tf.Tensor

def linear(layer_output_fixed):
    logits = tf.layers.dense(layer_output_fixed, 2)
    return logits


def mlp(layer_output_fixed):
    hidden = tf.layers.dense(layer_output_fixed, 768, activation='tanh')
    logits = tf.layers.dense(hidden, 2)
    return logits


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


def probability_mae(true_y, logits):
    pred_prob = tf.nn.softmax(logits, axis=-1)
    error = tf.abs(pred_prob - true_y)
    return error


def probe_modeling(layer_output, prob_v, input_mask, network, loss_fn=probability_mae):
    true_y = tf.stop_gradient(prob_v)
    layer_output_fixed = tf.stop_gradient(layer_output)
    logits = network(layer_output_fixed)
    per_token_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=true_y)
    per_token_losses = tf.reduce_sum(per_token_losses, axis=2)

    input_mask_float = tf.cast(input_mask, tf.float32)
    per_token_losses = per_token_losses * input_mask_float # [batch, seq_length]
    num_valid = tf.reduce_sum(input_mask_float, axis=1) + 1e-6
    error = loss_fn(true_y, logits)
    losses = tf.reduce_sum(per_token_losses, axis=1) / num_valid
    loss = tf.reduce_mean(losses)
    return PerLayerModel(loss, losses, logits, error)


class PerLayerModel(NamedTuple):
    loss: Tensor  # []
    losses: Tensor
    logits: Tensor  # [batch, seq_length, 2]
    error: Tensor  # [batch, seq_length]