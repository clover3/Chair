from typing import NamedTuple, Tuple

import tensorflow as tf
from keras.utils.losses_utils import ReductionV2

from models.transformer.bert_common_v2 import get_activation, create_initializer
from models.transformer.optimization_v2 import create_optimizer_from_config
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2
from tlm.model_cnfig import JsonConfig
from tlm.qtype.qtype_model_fn import set_dropout_to_zero, dummy_fn
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import log_features, log_var_assignments, get_tpu_scaffold_or_init
from tlm.training.train_config import TrainConfigEx


def get_alpha_from_config(model_config, default=1):
    try:
        alpha = model_config.alpha
    except Exception as e:
        print(e)
        alpha = default
    tf_logging.info("Using alpha of {}".format(alpha))
    return alpha


def get_mask_from_input_ids(input_ids):
    return tf.cast(tf.not_equal(input_ids, 0), tf.int32)


def process_feature_concat(features):
    def do_concat(input_ids1, segment_ids1, input_ids2, segment_ids2):
        input_mask1 = get_mask_from_input_ids(input_ids1)
        input_mask2 = get_mask_from_input_ids(input_ids2)
        return InputTriplet(input_ids=tf.concat([input_ids1, input_ids2], axis=0),
                            input_mask=tf.concat([input_mask1, input_mask2], axis=0),
                            segment_ids=tf.concat([segment_ids1, segment_ids2], axis=0))

    qe_inputs = do_concat(features["q_e_input_ids1"], features["q_e_segment_ids1"],
                          features["q_e_input_ids2"], features["q_e_segment_ids2"],
                          )
    de_inputs = do_concat(features["d_e_input_ids1"], features["d_e_segment_ids1"],
                          features["d_e_input_ids2"], features["d_e_segment_ids2"],
                          )
    return de_inputs, qe_inputs


class InputTriplet(NamedTuple):
    input_ids: tf.Tensor
    segment_ids: tf.Tensor
    input_mask: tf.Tensor


def process_feature(features) -> Tuple[InputTriplet, InputTriplet]:
    qe_inputs = InputTriplet(input_ids=features["q_e_input_ids"],
                             segment_ids=features["q_e_segment_ids"],
                             input_mask=get_mask_from_input_ids(features["q_e_input_ids"])
                             )
    de_inputs = InputTriplet(input_ids=features["d_e_input_ids"],
                             segment_ids=features["d_e_segment_ids"],
                             input_mask=get_mask_from_input_ids(features["d_e_input_ids"])
                             )
    return de_inputs, qe_inputs


def get_mse_loss_modeling(query_document_score, feature):
    label_ids = feature["label_ids"]
    losses = tf.keras.losses.MeanSquaredError(reduction=ReductionV2.NONE)(label_ids, query_document_score)
    loss = tf.reduce_mean(losses)
    return loss, losses, query_document_score


def get_mae_loss_modeling(query_document_score, feature):
    label_ids = feature["label_ids"]
    losses = tf.keras.losses.MeanAbsoluteError(reduction=ReductionV2.NONE)(label_ids, query_document_score)
    loss = tf.reduce_mean(losses)
    return loss, losses, query_document_score


def get_pairwise_loss_modeling(query_document_score, feature):
    paired_scores = tf.reshape(query_document_score, [2, -1])
    scores_pos = paired_scores[0]
    scores_neg = paired_scores[1]
    y_pred = scores_pos - scores_neg
    losses = tf.maximum(1.0 - y_pred, 0)
    loss = tf.reduce_mean(losses)
    return loss, losses, y_pred


def reshape_split(tensor):
    tensor_paired = tf.reshape(tensor, [2, -1, tensor.shape[1]])
    return tensor_paired[0], tensor_paired[1]


def qtype_modeling_single_mlp(config, pooled_output):
    h = tf.keras.layers.Dense(config.intermediate_size,
                              activation=get_activation(config.hidden_act))(pooled_output)
    h = tf.keras.layers.Dense(config.q_voca_size)(h)
    return h


def qde_metric(y_pred, qtype_vector1, qtype_vector2):
    """Computes the loss and accuracy of the model."""
    is_correct = tf.cast(tf.less(0.0, y_pred), tf.int32)
    acc = tf.compat.v1.metrics.accuracy(tf.ones_like(is_correct), is_correct)
    def get_l2_loss_per_inst(vector):
        return tf.reduce_sum(vector * vector / 2, axis=1)

    l2_loss = get_l2_loss_per_inst(qtype_vector1) + get_l2_loss_per_inst(qtype_vector2)

    return {
        "pairwise_acc": acc,
        "l2_loss": l2_loss,
    }


def model_fn_qde(FLAGS,
                 process_feature,
                 get_loss_modeling,
                 get_qtype_modeling,
                 special_flags=[]
                 ):
    model_config_o = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    use_one_hot_embeddings = FLAGS.use_tpu
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        log_features(features)
        de_inputs, qe_inputs = process_feature(features)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not is_training:
            model_config = set_dropout_to_zero(model_config_o)
        else:
            model_config = model_config_o

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model_1 = BertModel(
                config=model_config,
                is_training=is_training,
                input_ids=qe_inputs.input_ids,
                input_mask=qe_inputs.input_mask,
                token_type_ids=qe_inputs.segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled1 = model_1.get_pooled_output() # [batch_size * 2, hidden_size]
            qtype_vector1 = get_qtype_modeling(model_config, pooled1)  # [batch_size * 2, qtype_length]

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model_2 = BertModel(
                config=model_config,
                is_training=is_training,
                input_ids=de_inputs.input_ids,
                input_mask=de_inputs.input_mask,
                token_type_ids=de_inputs.segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled2 = model_2.get_pooled_output()
            qtype_vector2 = get_qtype_modeling(model_config, pooled2)

        query_document_score = tf.reduce_sum(tf.multiply(qtype_vector1, qtype_vector2), axis=1)
        if "bias" in special_flags:
            tf_logging.info("Using bias")
            bias = tf.Variable(initial_value=0.0, trainable=True)
            query_document_score = query_document_score + bias

        loss, losses, y_pred = get_loss_modeling(query_document_score, features)

        if "l2_loss" in special_flags:
            tf_logging.info("Using l2_loss")
            l2_loss_1 = tf.nn.l2_loss(qtype_vector1)
            l2_loss_2 = tf.nn.l2_loss(qtype_vector2)
            alpha = get_alpha_from_config(model_config, 1)
            l2_loss_total = (l2_loss_1 + l2_loss_2) * alpha
            loss += l2_loss_total

        if "paired_pred" in special_flags:
            qtype_vector_qe1, qtype_vector_qe2 = reshape_split(qtype_vector1)
            qtype_vector_de1, qtype_vector_de2 = reshape_split(qtype_vector2)
            prediction = {
                "data_id": features["data_id"],
                "qtype_vector_qe1": qtype_vector_qe1,
                "qtype_vector_qe2": qtype_vector_qe2,
                "qtype_vector_de1": qtype_vector_de1,
                "qtype_vector_de2": qtype_vector_de2,
            }
        else:
            prediction = {
                "data_id": features["data_id"],
                "label_ids": features["label_ids"],
                # "qe_input_ids": qe_input_ids,
                # "de_input_ids": de_input_ids,
                # "qtype_vector_qe": qtype_vector1,
                # "qtype_vector_de": qtype_vector2,
                "logits": query_document_score,
            }

        all_tvars = tf.compat.v1.trainable_variables()
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, all_tvars)
            log_var_assignments(all_tvars, initialized_variable_names)
        else:
            init_fn = dummy_fn
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        optimizer_factory = lambda x: create_optimizer_from_config(x, train_config)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_logging.info("Using single lr ")
            train_op = optimizer_factory(loss)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (qde_metric, [
                y_pred, qtype_vector1, qtype_vector2
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics,
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


    return model_fn


def model_fn_qde3(FLAGS,
                 process_feature,
                 get_loss_modeling,
                 get_qtype_modeling,
                 ):
    def single_bias_model(config, vector):
        dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh,
                                      kernel_initializer=create_initializer(config.initializer_range))
        return dense(vector)

    special_flags = FLAGS.special_flags.split(",")
    model_config_o = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    use_one_hot_embeddings = FLAGS.use_tpu
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        log_features(features)
        de_inputs, qe_inputs = process_feature(features)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not is_training:
            model_config = set_dropout_to_zero(model_config_o)
        else:
            model_config = model_config_o

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model_1 = BertModel(
                config=model_config,
                is_training=is_training,
                input_ids=qe_inputs.input_ids,
                input_mask=qe_inputs.input_mask,
                token_type_ids=qe_inputs.segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled1 = model_1.get_pooled_output() # [batch_size * 2, hidden_size]
            qtype_vector1 = get_qtype_modeling(model_config, pooled1)  # [batch_size * 2, qtype_length]
            q_bias = single_bias_model(model_config, pooled1)

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model_2 = BertModel(
                config=model_config,
                is_training=is_training,
                input_ids=de_inputs.input_ids,
                input_mask=de_inputs.input_mask,
                token_type_ids=de_inputs.segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled2 = model_2.get_pooled_output()
            qtype_vector2 = get_qtype_modeling(model_config, pooled2)
            d_bias = single_bias_model(model_config, pooled2)

        query_document_score = tf.reduce_sum(tf.multiply(qtype_vector1, qtype_vector2), axis=1)
        bias = tf.Variable(initial_value=0.0, trainable=True)
        query_document_score += query_document_score + bias
        query_document_score += q_bias
        query_document_score += d_bias

        query_document_score1 = tf.reduce_sum(tf.multiply(qtype_vector1, qtype_vector2), axis=1)
        bias = tf.Variable(initial_value=0.0, trainable=True)
        query_document_score2 = query_document_score1 + bias
        query_document_score3 = query_document_score2 + q_bias
        query_document_score = query_document_score3 + d_bias
        bias2 = query_document_score2 -query_document_score1
        alpha = get_alpha_from_config(model_config, 1)
        loss, losses, y_pred = get_loss_modeling(query_document_score, features)
        tensor_dict = {
            "data_id": features["data_id"],
            "label_ids": features["label_ids"],
            "qtype_vector_qe": qtype_vector1,
            "qtype_vector_de": qtype_vector2,
            "de_input_ids": de_inputs.input_ids,
            "logits": query_document_score,
            "bias": bias2,
            "q_bias": q_bias,
            "d_bias": d_bias
        }
        if "predict_vector" in special_flags:
            tensor_to_predict = ["data_id", "label_ids", "logits", "de_input_ids",
                                 "qtype_vector_qe", "qtype_vector_de", "q_bias", "d_bias", "bias"]
        else:
            tensor_to_predict = ["data_id", "label_ids", "logits"]
        prediction = {k: tensor_dict[k] for k in tensor_to_predict}

        sparsity_loss_1 = get_l1_loss_w(qtype_vector1)
        sparsity_loss_2 = get_l1_loss_w(qtype_vector2)
        sparsity_loss_total = (sparsity_loss_1 + sparsity_loss_2) * alpha
        loss += tf.reduce_mean(sparsity_loss_total)

        all_tvars = tf.compat.v1.trainable_variables()
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, all_tvars)
            log_var_assignments(all_tvars, initialized_variable_names)
        else:
            init_fn = dummy_fn
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        optimizer_factory = lambda x: create_optimizer_from_config(x, train_config)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_logging.info("Using single lr ")
            train_op = optimizer_factory(loss)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (qde_metric, [
                y_pred, sparsity_loss_total
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics,
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



    return model_fn


def get_l1_loss_w(vector):
    abs_vector = tf.abs(vector)
    max_val = tf.reduce_max(abs_vector, axis=1)
    losses = tf.reduce_sum(abs_vector, axis=1) - max_val
    return losses



def qde4_metric(y_pred, losses):
    """Computes the loss and accuracy of the model."""
    is_correct = tf.cast(tf.less(0.0, y_pred), tf.int32)
    acc = tf.compat.v1.metrics.accuracy(tf.ones_like(is_correct), is_correct)
    return {
        "pairwise_acc": acc,
        "sparsity_loss": losses,
    }


def model_fn_qde4(FLAGS,
                  process_feature,
                  get_loss_modeling,
                  get_qtype_modeling,
                  ):
    def single_bias_model(config, vector):
        dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh,
                                      kernel_initializer=create_initializer(config.initializer_range))
        v = dense(vector)
        return tf.reshape(v, [-1])

    model_config_o = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    use_one_hot_embeddings = FLAGS.use_tpu
    special_flags = FLAGS.special_flags.split(",")

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        log_features(features)
        de_inputs, qe_inputs = process_feature(features)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not is_training:
            model_config = set_dropout_to_zero(model_config_o)
        else:
            model_config = model_config_o

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model_1 = BertModel(
                config=model_config,
                is_training=is_training,
                input_ids=qe_inputs.input_ids,
                input_mask=qe_inputs.input_mask,
                token_type_ids=qe_inputs.segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled1 = model_1.get_pooled_output() # [batch_size * 2, hidden_size]
            qtype_vector1 = get_qtype_modeling(model_config, pooled1)  # [batch_size * 2, qtype_length]
            q_bias = single_bias_model(model_config, pooled1)

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model_2 = BertModel(
                config=model_config,
                is_training=is_training,
                input_ids=de_inputs.input_ids,
                input_mask=de_inputs.input_mask,
                token_type_ids=de_inputs.segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled2 = model_2.get_pooled_output()
            qtype_vector2 = get_qtype_modeling(model_config, pooled2)
            d_bias = single_bias_model(model_config, pooled2)

        query_document_score1 = tf.reduce_sum(tf.multiply(qtype_vector1, qtype_vector2), axis=1)
        bias = tf.Variable(initial_value=0.0, trainable=True)
        query_document_score2 = query_document_score1 + bias
        query_document_score3 = query_document_score2 + q_bias
        query_document_score = query_document_score3 + d_bias
        bias2 = query_document_score2 -query_document_score1
        alpha = get_alpha_from_config(model_config, 1)
        loss, losses, y_pred = get_loss_modeling(query_document_score, features)
        tensor_dict = {
            "data_id": features["data_id"],
            "label_ids": features["label_ids"],
            "qtype_vector_qe": qtype_vector1,
            "qtype_vector_de": qtype_vector2,
            "de_input_ids": de_inputs.input_ids,
            "logits": query_document_score,
            "bias": bias2,
            "q_bias": q_bias,
            "d_bias": d_bias
        }
        if "predict_vector" in special_flags:
            tensor_to_predict = ["data_id", "label_ids", "logits", "de_input_ids",
                                 "qtype_vector_qe", "qtype_vector_de", "q_bias", "d_bias", "bias"]
        else:
            tensor_to_predict = ["data_id", "label_ids", "logits"]
        prediction = {k: tensor_dict[k] for k in tensor_to_predict}

        sparsity_loss_1 = get_l1_loss_w(qtype_vector1)
        # sparsity_loss_2 = qtype_vector2
        sparsity_loss_total = sparsity_loss_1 * alpha
        loss += tf.reduce_mean(sparsity_loss_total)

        all_tvars = tf.compat.v1.trainable_variables()
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, all_tvars)
            log_var_assignments(all_tvars, initialized_variable_names)
        else:
            init_fn = dummy_fn
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        optimizer_factory = lambda x: create_optimizer_from_config(x, train_config)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_logging.info("Using single lr ")
            train_op = optimizer_factory(loss)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (qde_metric, [
                y_pred, sparsity_loss_total
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics,
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

    return model_fn



def model_fn_qde5(FLAGS,
                  process_feature,
                  get_loss_modeling,
                  get_qtype_modeling,
                  ):
    def single_bias_model(config, vector):
        dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh,
                                      kernel_initializer=create_initializer(config.initializer_range))
        return dense(vector)

    model_config_o = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    use_one_hot_embeddings = FLAGS.use_tpu
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        log_features(features)
        de_inputs, qe_inputs = process_feature(features)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not is_training:
            model_config = set_dropout_to_zero(model_config_o)
        else:
            model_config = model_config_o

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            qe_input_ids, qe_input_mask, qe_segment_ids = qe_inputs
            model_1 = BertModel(
                config=model_config,
                is_training=is_training,
                input_ids=qe_inputs.input_ids,
                input_mask=qe_inputs.input_mask,
                token_type_ids=qe_inputs.segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled1 = model_1.get_pooled_output() # [batch_size * 2, hidden_size]
            qtype_vector1 = get_qtype_modeling(model_config, pooled1)  # [batch_size * 2, qtype_length]
            q_bias = single_bias_model(model_config, pooled1)

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model_2 = BertModel(
                config=model_config,
                is_training=is_training,
                input_ids=de_inputs.input_ids,
                input_mask=de_inputs.input_mask,
                token_type_ids=de_inputs.segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled2 = model_2.get_pooled_output()
            qtype_vector2 = get_qtype_modeling(model_config, pooled2)
            d_bias = single_bias_model(model_config, pooled2)

        query_document_score = tf.reduce_sum(tf.multiply(qtype_vector1, qtype_vector2), axis=1)
        bias = tf.Variable(initial_value=0.0, trainable=True)
        query_document_score += query_document_score
        try:

            alpha = model_config.alpha
        except Exception as e:
            print(e)
            alpha = 1
        tf_logging.info("Using alpha of {}".format(alpha))

        loss, losses, y_pred = get_loss_modeling(query_document_score, features)
        sparsity_loss_1 = get_l1_loss_w(qtype_vector1)
        # sparsity_loss_2 = qtype_vector2
        sparsity_loss_total = sparsity_loss_1 * alpha
        loss += tf.reduce_mean(sparsity_loss_total)
        prediction = {
            "data_id": features["data_id"],
            "label_ids": features["label_ids"],
            # "qe_input_ids": qe_input_ids,
            # "de_input_ids": de_input_ids,
            # "qtype_vector_qe": qtype_vector1,
            # "qtype_vector_de": qtype_vector2,
            "logits": query_document_score,
        }

        all_tvars = tf.compat.v1.trainable_variables()
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, all_tvars)
            log_var_assignments(all_tvars, initialized_variable_names)
        else:
            init_fn = dummy_fn
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        optimizer_factory = lambda x: create_optimizer_from_config(x, train_config)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_logging.info("Using single lr ")
            train_op = optimizer_factory(loss)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (qde_metric, [
                y_pred, sparsity_loss_total
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics,
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


    return model_fn