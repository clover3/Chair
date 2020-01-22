import tensorflow as tf

import tlm.model.base as modeling
from models.transformer.bert_common_v2 import dropout
from models.transformer.optimization_v2 import create_optimizer_from_config
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.training import assignment_map
from tlm.training.model_fn_common import log_features, align_checkpoint, get_tpu_scaffold_or_init, log_var_assignments
from tlm.training.train_config import TrainConfig


class PairWise:
    pass

def combine_paired_input_features(features):
    input_ids1 = features["input_ids1"]
    input_mask1 = features["input_mask1"]
    segment_ids1 = features["segment_ids1"]

    # Negative Example
    input_ids2 = features["input_ids2"]
    input_mask2 = features["input_mask2"]
    segment_ids2 = features["segment_ids2"]

    input_ids = tf.concat([input_ids1, input_ids2], axis=0)
    input_mask = tf.concat([input_mask1, input_mask2], axis=0)
    segment_ids = tf.concat([segment_ids1, segment_ids2], axis=0)
    return input_ids, input_mask, segment_ids


def pairwise_model(pooled_output):
    logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output)
    pair_logits = tf.reshape(logits, [2, -1])
    y_pred = pair_logits[0, :] - pair_logits[1, :]
    losses = tf.maximum(1.0 - y_pred, 0)
    loss = tf.reduce_mean(losses)
    return loss, losses, y_pred


def pairwise_cross_entropy(pooled_output):
    logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output)
    pair_logits = tf.reshape(logits, [2, -1])
    prob = tf.nn.softmax(pair_logits, axis=0)
    losses = 1 - prob[0, :]
    loss = tf.reduce_mean(losses)
    return loss, losses, prob[0, :]


def cross_entropy(pooled_output):
    logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled_output)
    real_batch_size = tf.cast(logits.shape[0] / 2, tf.int32)

    labels = tf.concat([tf.ones([real_batch_size], tf.int32),
                         tf.zeros([real_batch_size], tf.int32)], axis=0)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels)
    loss = tf.reduce_mean(losses)
    return loss, losses, tf.reshape(logits, [2, -1, 2])[0, :, 1]


def checkpoint_init(assignment_fn, train_config):
    tvars = tf.compat.v1.trainable_variables()
    initialized_variable_names, init_fn = align_checkpoint(tvars, train_config.init_checkpoint, assignment_fn)
    scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
    log_var_assignments(tvars, initialized_variable_names)
    return scaffold_fn


def ranking_estimator_spec(mode, loss, losses, y_pred, scaffold_fn, optimizer_factory, prediction=None):
    TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf_logging.info("Using single lr ")
        train_op = optimizer_factory(loss)
        output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
        output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=None,
                                       scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        if prediction is None:
            prediction = {
                "y_pred": y_pred,
                "losses": losses,
            }
        output_spec = TPUEstimatorSpec(mode=mode, loss=loss,
                                       predictions=prediction,
                                       scaffold_fn=scaffold_fn)
    else:
        assert False
    return output_spec


def apply_loss_modeling(modeling_opt, pooled_output):
    if modeling_opt == "hinge":
        loss, losses, y_pred = pairwise_model(pooled_output)
    elif modeling_opt == "pair_ce":
        loss, losses, y_pred = pairwise_cross_entropy(pooled_output)
    elif modeling_opt == "ce":
        loss, losses, y_pred = cross_entropy(pooled_output)
    elif modeling_opt == "all_pooling":

        loss, losses, y_pred = cross_entropy(pooled_output)
    else:
        assert False
    return loss, losses, y_pred


def model_fn_ranking(FLAGS):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    train_config = TrainConfig.from_flags(FLAGS)
    modeling_opt = FLAGS.modeling

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_ranking")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids, input_mask, segment_ids = combine_paired_input_features(features)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # Updated

        model = BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )
        pooled_output = model.get_pooled_output()
        if is_training:
            pooled_output = dropout(pooled_output, 0.1)

        loss, losses, y_pred = apply_loss_modeling(modeling_opt, pooled_output)


        assignment_fn = assignment_map.get_bert_assignment_map
        scaffold_fn = checkpoint_init(assignment_fn, train_config)

        optimizer_factory = lambda x: create_optimizer_from_config(x, train_config)
        input_ids1 = tf.identity(features["input_ids1"])
        input_ids2 = tf.identity(features["input_ids2"])
        prediction = {
            "input_ids1": input_ids1,
            "input_ids2": input_ids2
        }
        return ranking_estimator_spec(mode, loss, losses, y_pred, scaffold_fn, optimizer_factory, prediction)


    return model_fn

def rank_predict_estimator_spec(logits, mode, scaffold_fn):
    TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
    if mode == tf.estimator.ModeKeys.PREDICT:
        output_spec = TPUEstimatorSpec(mode=mode, predictions={"logits": logits},
                                       scaffold_fn=scaffold_fn)
    else:
        assert False
    return output_spec


def get_prediction_structure(modeling_opt, pooled_output):
    if modeling_opt == "hinge" or modeling_opt == "pair_ce":
        logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output)
    elif modeling_opt == "ce":
        raw_logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled_output)
        probs = tf.nn.softmax(raw_logits, axis=1)
        logits = probs[:, 1]
    else:
        assert False
    return logits


def model_fn_rank_pred(FLAGS):
    train_config = TrainConfig.from_flags(FLAGS)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    modeling_opt = FLAGS.modeling

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_sero_classification")
        """The `model_fn` for TPUEstimator."""
        log_features(features)
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # Updated
        model = BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )
        pooled_output = model.get_pooled_output()
        if is_training:
            pooled_output = dropout(pooled_output, 0.1)

        logits = get_prediction_structure(modeling_opt, pooled_output)
        loss = 0

        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = assignment_map.get_bert_assignment_map
        initialized_variable_names, init_fn = align_checkpoint(tvars, train_config.init_checkpoint, assignment_fn)
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)

        output_spec = rank_predict_estimator_spec(logits, mode, scaffold_fn)
        return output_spec

    return model_fn

