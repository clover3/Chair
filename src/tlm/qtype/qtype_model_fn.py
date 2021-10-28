import copy

import tensorflow as tf

from models.transformer.optimization_v2 import create_optimizer_from_config
from tf_util.tf_logging import tf_logging
from tlm.model_cnfig import JsonConfig
from tlm.qtype.BertQType import BertQType
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import log_features, get_tpu_scaffold_or_init
from tlm.training.ranking_model_common import combine_paired_input_features, get_prediction_structure
from tlm.training.ranking_model_fn import ranking_estimator_spec
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


def model_fn_qtype(FLAGS):
    model_config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    modeling_opt = FLAGS.modeling

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_ranking")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids, input_mask, segment_ids = combine_paired_input_features(features)
        drop_input_ids, drop_input_mask, drop_segment_ids = combine_paired_input_features(features)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model_config_predict = copy.deepcopy(model_config)
        # Updated
        model_config_predict.hidden_dropout_prob = 0.0
        model_config_predict.attention_probs_dropout_prob = 0.0
        model = BertQType()
        sep_id = 101
        with tf.compat.v1.variable_scope("SCOPE1"):
            all_layers_seg1 = model.build_tower1(model_config_predict, is_training,
                                                 input_ids, input_mask, segment_ids,
                                                 train_config.use_one_hot_embeddings)
            pooled_output = model.get_pooled_output()
            orig_logits = get_prediction_structure(modeling_opt, pooled_output)
            orig_probs = tf.nn.softmax(orig_logits)
        with tf.compat.v1.variable_scope("SCOPE2"):
            model.build_tower2(model_config, model_config_predict, all_layers_seg1,
                               drop_input_ids, drop_input_mask, drop_segment_ids,
                               is_training, sep_id, train_config.use_one_hot_embeddings)
            drop_pooled_output = model.drop_pooled_output
            drop_logits = get_prediction_structure(modeling_opt, drop_pooled_output)
            drop_probs = tf.nn.softmax(drop_logits)

        kl = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)
        losses = kl(orig_probs, drop_probs)
        tvars = tf.compat.v1.trainable_variables()
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, tvars)
        else:
            def init_fn():
                pass
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        optimizer_factory = lambda x: create_optimizer_from_config(x, train_config)
        input_ids1 = tf.identity(features["input_ids1"])
        input_ids2 = tf.identity(features["input_ids2"])
        prediction = {
            "input_ids1": input_ids1,
            "input_ids2": input_ids2
        }
        loss = tf.reduce_mean(losses)
        return ranking_estimator_spec(mode, loss, losses, drop_logits, scaffold_fn, optimizer_factory, prediction)
    return model_fn
