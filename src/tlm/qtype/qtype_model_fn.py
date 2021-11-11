import copy

import tensorflow as tf

from models.transformer.optimization_v2 import create_optimizer_from_config
from tf_util.tf_logging import tf_logging
from tlm.model_cnfig import JsonConfig
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import log_features, get_tpu_scaffold_or_init, log_var_assignments
from tlm.training.ranking_model_common import combine_paired_input_features, get_prediction_structure
from tlm.training.train_config import TrainConfigEx


def combine_paired_input_features_drop(features):
    input_ids1 = features["drop_input_ids1"]
    input_mask1 = features["drop_input_mask1"]
    segment_ids1 = features["drop_segment_ids1"]

    # Negative Example
    input_ids2 = features["drop_input_ids2"]
    input_mask2 = features["drop_input_mask2"]
    segment_ids2 = features["drop_segment_ids2"]

    input_ids = tf.concat([input_ids1, input_ids2], axis=0)
    input_mask = tf.concat([input_mask1, input_mask2], axis=0)
    segment_ids = tf.concat([segment_ids1, segment_ids2], axis=0)
    return input_ids, input_mask, segment_ids


def get_task_loss(pair_logits):
    y_pred = pair_logits[:, 0] - pair_logits[:, 1]
    losses = tf.maximum(1.0 - y_pred, 0)

    is_correct = tf.cast(tf.less(0.0, y_pred), tf.int32)
    acc = tf.compat.v1.metrics.accuracy(tf.ones_like(is_correct), is_correct)
    loss = tf.compat.v1.metrics.mean_absolute_error(tf.zeros_like(losses), losses)
    return acc, loss


def metric_fn_qtype(orig_input_mask, drop_input_mask, orig_logits_pair, drop_logits_pair, qtype_weights_paired):
    mae_metric = tf.compat.v1.metrics.mean_absolute_error
    """Computes the loss and accuracy of the model."""
    num_tokens_orig = tf.reduce_sum(orig_input_mask, axis=1)
    num_tokens_drop = tf.reduce_sum(drop_input_mask, axis=1)

    num_diff_tokens = mae_metric(num_tokens_orig, num_tokens_drop)
    logit_mae = mae_metric(orig_logits_pair, drop_logits_pair)

    acc_orig, loss_orig = get_task_loss(orig_logits_pair)
    acc_drop, loss_drop = get_task_loss(drop_logits_pair)
    diff = qtype_weights_paired[:, 0, :] - qtype_weights_paired[:, 1, :]
    query_consistency = tf.reduce_sum(diff * diff, axis=1)
    query_consistency_metric = mae_metric(query_consistency, tf.zeros_like(query_consistency))
    return {
        "num_diff_tokens": num_diff_tokens,
        "logit_mae": logit_mae,
        "acc_orig": acc_orig,
        "loss_orig": loss_orig,
        "acc_drop": acc_drop,
        "loss_drop": loss_drop,
        "loss_query_consistency": query_consistency_metric
    }


def pairing_reshape(logits):
    head_paired = tf.reshape(logits, [2, -1])
    tail_paired = tf.transpose(head_paired, [1, 0])
    return tail_paired

def dummy_fn():
    pass


def set_dropout_to_zero(model_config):
    model_config_predict = copy.deepcopy(model_config)
    # Updated
    model_config_predict.hidden_dropout_prob = 0.0
    model_config_predict.attention_probs_dropout_prob = 0.0
    return model_config_predict


def model_fn_qtype_pairwise(FLAGS, model_class, qtype_options=[]):
    model_config_o = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    modeling_opt = FLAGS.modeling

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        log_features(features)
        input_ids, input_mask, segment_ids = combine_paired_input_features(features)
        drop_input_ids, drop_input_mask, drop_segment_ids = combine_paired_input_features_drop(features)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not is_training:
            model_config = set_dropout_to_zero(model_config_o)
        else:
            model_config = model_config_o

        model_config_predict = set_dropout_to_zero(model_config_o)
        model = model_class()
        sep_id = 101
        with tf.compat.v1.variable_scope("SCOPE1"):
            all_layers_seg1 = model.build_tower1(model_config_predict, is_training,
                                                 input_ids, input_mask, segment_ids,
                                                 train_config.use_one_hot_embeddings)
            pooled_output = model.get_pooled_output()
            orig_logits = get_prediction_structure(modeling_opt, pooled_output)
        with tf.compat.v1.variable_scope("SCOPE2"):
            model.build_tower2(model_config, model_config_predict, all_layers_seg1,
                               drop_input_ids, drop_input_mask, drop_segment_ids,
                               is_training, sep_id, train_config.use_one_hot_embeddings)
            drop_pooled_output = model.drop_pooled_output
            drop_logits = get_prediction_structure(modeling_opt, drop_pooled_output)

            qtype_weights = model.q_embedding_model.get_qtype_weights()

        if "pair_loss" in qtype_options:
            tf_logging.info("Use pair loss")
            y_diff = drop_logits[0, :] - drop_logits[1, :]
            losses = tf.maximum(1.0 - y_diff, 0)
        else:
            losses = tf.keras.losses.MAE(orig_logits, drop_logits)

        all_tvars = tf.compat.v1.trainable_variables()
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, all_tvars)
            log_var_assignments(all_tvars, initialized_variable_names)
        else:
            init_fn = dummy_fn

        orig_logits_pair = pairing_reshape(orig_logits)
        drop_logits_pair = pairing_reshape(drop_logits)
        tvars = []
        for v in all_tvars:
            if "qtype_modeling" in v.name:
                tvars.append(v)

        tf_logging.info("There are {} trainable variables".format(len(tvars)))
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        optimizer_factory = lambda x: create_optimizer_from_config(x, train_config, tvars)
        input_ids1 = tf.identity(features["input_ids1"])
        input_ids2 = tf.identity(features["input_ids2"])
        drop_input_ids1 = tf.identity(features["drop_input_ids1"])
        drop_input_ids2 = tf.identity(features["drop_input_ids2"])
        qtype_weights_paired = tf.reshape(qtype_weights, [2, -1, model_config.q_voca_size])
        qtype_weights_paired = tf.transpose(qtype_weights_paired, [1, 0, 2])
        qtype_raw_logits = model.q_embedding_model.get_qtype_raw_logits()
        qtype_raw_logits_paired = tf.reshape(qtype_raw_logits, [2, -1, model_config.q_voca_size])

        query_consistency = tf.nn.l2_loss(qtype_raw_logits_paired[0, :, :]
                                          - qtype_raw_logits_paired[1, :, :])

        prediction = {
            "input_ids1": input_ids1,
            "input_ids2": input_ids2,
            "drop_input_ids1": drop_input_ids1,
            "drop_input_ids2": drop_input_ids2,
            "orig_logits_pair": orig_logits_pair,
            "drop_logits_pair": drop_logits_pair,
            "qtype_weights_paired": qtype_weights_paired,
        }
        factor = 20
        loss = tf.reduce_mean(losses) + query_consistency * factor

        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_logging.info("Using single lr ")
            train_op = optimizer_factory(loss)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn_qtype, [
                input_mask, drop_input_mask, orig_logits_pair, drop_logits_pair, qtype_weights_paired
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics,
                                           scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            if prediction is None:
                prediction = {
                    "y_pred": drop_logits,
                    "losses": losses,
                }
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss,
                                           predictions=prediction,
                                           scaffold_fn=scaffold_fn)
        else:
            assert False
        return output_spec

    return model_fn


