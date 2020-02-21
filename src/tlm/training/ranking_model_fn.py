import tensorflow as tf

import tlm.model.base as modeling
from models.transformer.bert_common_v2 import dropout
from models.transformer.optimization_v2 import create_optimizer_from_config
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.training import assignment_map
from tlm.training.model_fn_common import log_features, align_checkpoint, get_tpu_scaffold_or_init, log_var_assignments
from tlm.training.ranking_model_common import combine_paired_input_features, get_prediction_structure, \
    apply_loss_modeling
from tlm.training.train_config import TrainConfig


class PairWise:
    pass


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

        loss, losses, y_pred = apply_loss_modeling(modeling_opt, pooled_output, features)


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



def rank_predict_estimator_spec(logits, mode, scaffold_fn, predictions=None):
    TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
    if mode == tf.estimator.ModeKeys.PREDICT:
        if predictions is None:
            predictions ={"logits": logits}
        output_spec = TPUEstimatorSpec(mode=mode, predictions=predictions,
                                       scaffold_fn=scaffold_fn)
    else:
        assert False
    return output_spec


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

        predictions = None
        if modeling_opt == "multi_label_hinge":
            predictions = {
                "input_ids":input_ids,
                "logits":logits,
            }
        output_spec = rank_predict_estimator_spec(logits, mode, scaffold_fn, predictions)
        return output_spec

    return model_fn

