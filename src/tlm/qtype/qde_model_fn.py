import tensorflow as tf
from keras.utils.losses_utils import ReductionV2
from tensorflow_core.python.ops.gen_nn_ops import l2_loss

from models.transformer.bert_common_v2 import get_activation
from models.transformer.optimization_v2 import create_optimizer_from_config
from tlm.model.base import BertModel
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2
from tlm.model_cnfig import JsonConfig
from tlm.qtype.qtype_model_fn import set_dropout_to_zero, dummy_fn
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import log_features, log_var_assignments, get_tpu_scaffold_or_init
from tlm.training.ranking_model_fn import ranking_estimator_spec
from tlm.training.train_config import TrainConfigEx


def get_mask_from_input_ids(input_ids):
    return tf.cast(tf.not_equal(input_ids, 0), tf.int32)


def process_feature_concat(features):
    def do_concat(input_ids1, segment_ids1, input_ids2, segment_ids2):
        input_mask1 = get_mask_from_input_ids(input_ids1)
        input_mask2 = get_mask_from_input_ids(input_ids2)
        return (tf.concat([input_ids1, input_ids2], axis=0),
                tf.concat([input_mask1, input_mask2], axis=0),
                tf.concat([segment_ids1, segment_ids2], axis=0))

    qe_inputs = do_concat(features["q_e_input_ids1"], features["q_e_segment_ids1"],
                          features["q_e_input_ids2"], features["q_e_segment_ids2"],
                          )
    de_inputs = do_concat(features["d_e_input_ids1"], features["d_e_segment_ids1"],
                          features["d_e_input_ids2"], features["d_e_segment_ids2"],
                          )
    return de_inputs, qe_inputs


def process_feature(features):
    qe_inputs = features["q_e_input_ids"], \
                features["q_e_segment_ids"], \
                get_mask_from_input_ids(features["q_e_input_ids"])
    de_inputs = features["d_e_input_ids"], \
                features["d_e_segment_ids"], \
                get_mask_from_input_ids(features["d_e_input_ids"])
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
            qe_input_ids, qe_input_mask, qe_segment_ids = qe_inputs
            model_1 = BertModel(
                config=model_config,
                is_training=is_training,
                input_ids=qe_input_ids,
                input_mask=qe_input_mask,
                token_type_ids=qe_segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled1 = model_1.get_pooled_output() # [batch_size * 2, hidden_size]
            qtype_vector1 = get_qtype_modeling(model_config, pooled1)  # [batch_size * 2, qtype_length]

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            de_input_ids, de_input_mask, de_segment_ids = de_inputs
            model_2 = BertModel(
                config=model_config,
                is_training=is_training,
                input_ids=de_input_ids,
                input_mask=de_input_mask,
                token_type_ids=de_segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled2 = model_2.get_pooled_output()
            qtype_vector2 = get_qtype_modeling(model_config, pooled2)

        query_document_score = tf.reduce_sum(tf.multiply(qtype_vector1, qtype_vector2), axis=1)
        loss, losses, y_pred = get_loss_modeling(query_document_score, features)
        if "l2_loss" in special_flags:
            loss += l2_loss(qtype_vector1)
            loss += l2_loss(qtype_vector2)

        if "paired_pred" in special_flags:
            qtype_vector_qe1, qtype_vector_qe2 = reshape_split(qtype_vector1)
            qtype_vector_de1, qtype_vector_de2 = reshape_split(qtype_vector2)
            qe_input_ids1, qe_input_ids2 = reshape_split(qe_input_ids)
            de_input_ids1, de_input_ids2 = reshape_split(de_input_ids)
            prediction = {
                "data_id": features["data_id"],
                "qe_input_ids1": qe_input_ids1,
                "qe_input_ids2": qe_input_ids2,
                "de_input_ids1": de_input_ids1,
                "de_input_ids2": de_input_ids2,
                "qtype_vector_qe1": qtype_vector_qe1,
                "qtype_vector_qe2": qtype_vector_qe2,
                "qtype_vector_de1": qtype_vector_de1,
                "qtype_vector_de2": qtype_vector_de2,
            }
        else:
            prediction = {
                "data_id": features["data_id"],
                "qe_input_ids": qe_input_ids,
                "de_input_ids": de_input_ids,
                "qtype_vector_qe": qtype_vector1,
                "qtype_vector_de": qtype_vector2,
            }

        all_tvars = tf.compat.v1.trainable_variables()
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, all_tvars)
            log_var_assignments(all_tvars, initialized_variable_names)
        else:
            init_fn = dummy_fn
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        optimizer_factory = lambda x: create_optimizer_from_config(x, train_config)
        return ranking_estimator_spec(mode, loss, losses, y_pred, scaffold_fn, optimizer_factory, prediction)

    return model_fn