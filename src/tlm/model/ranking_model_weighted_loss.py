import tensorflow as tf

import tlm.model.base as modeling
from models.transformer.bert_common_v2 import dropout
from models.transformer.optimization_v2 import create_optimizer_from_config
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.model_cnfig import JsonConfig
from tlm.training import assignment_map
from tlm.training.model_fn_common import log_features
from tlm.training.ranking_model_common import combine_paired_input_features
from tlm.training.ranking_model_fn import checkpoint_init, ranking_estimator_spec
from tlm.training.train_config import TrainConfigEx


def get_gradient_adjust_layer(factor):
    @tf.custom_gradient
    def gradient_adjust_layer(x):
        y = x

        def grad(dy):
            g_raw = dy * factor
            return g_raw

        return y, grad
    return gradient_adjust_layer


def get_random_gradient_drop_layer(factor):
    @tf.custom_gradient
    def gradient_adjust_layer(x):
        y = x

        def grad(dy):
            r_val = tf.random.uniform([], 0, 1)
            mask = tf.cast(tf.less(r_val, factor), tf.float32)
            g_raw = dy * mask
            return g_raw

        return y, grad
    return gradient_adjust_layer


def get_masked_gradient_drop_layer(mask):
    @tf.custom_gradient
    def gradient_adjust_layer(x):
        y = x

        def grad(dy):
            g_raw = dy * tf.cast(mask, tf.float32)
            return g_raw

        return y, grad
    return gradient_adjust_layer


def pairwise_model(features, factor, modeling_opt, pooled_output):
    if modeling_opt == "hinge":
        logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output)
        pair_logits = tf.reshape(logits, [2, -1])

        gradient_adjust_layer = get_gradient_adjust_layer(factor)
        logit_pos = gradient_adjust_layer(pair_logits[0, :])
        logit_neg = pair_logits[1, :]
        y_pred = logit_pos - logit_neg
        losses = tf.maximum(1.0 - y_pred, 0)
        loss = tf.reduce_mean(losses)
    elif modeling_opt == "random_drop":
        logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output)
        pair_logits = tf.reshape(logits, [2, -1])

        gradient_adjust_layer = get_random_gradient_drop_layer(factor)
        logit_pos = gradient_adjust_layer(pair_logits[0, :])
        logit_neg = pair_logits[1, :]
        y_pred = logit_pos - logit_neg
        losses = tf.maximum(1.0 - y_pred, 0)
        loss = tf.reduce_mean(losses)
    elif modeling_opt == "random_drop2":
        batch_size2, hidden_size = modeling.get_shape_list(pooled_output, expected_rank=2)
        pooled_output_3d = tf.reshape(pooled_output, [2, -1, hidden_size])
        pos_pooled = pooled_output_3d[0, :, :]
        neg_pooled = pooled_output_3d[1, :, :]
        gradient_adjust_layer = get_random_gradient_drop_layer(factor)
        pos_pooled = gradient_adjust_layer(pos_pooled)
        pooled_output_modified = tf.concat([pos_pooled, neg_pooled], axis=0)
        logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output_modified)
        pair_logits = tf.reshape(logits, [2, -1])
        logit_pos = pair_logits[0, :]
        logit_neg = pair_logits[1, :]
        y_pred = logit_pos - logit_neg
        losses = tf.maximum(1.0 - y_pred, 0)
        loss = tf.reduce_mean(losses)
    elif modeling_opt == "mask_drop":
        logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output)
        pair_logits = tf.reshape(logits, [2, -1])
        use_pos = features["use_pos"]
        drop_mask = tf.reshape(1 - use_pos, [-1])
        gradient_adjust_layer = get_masked_gradient_drop_layer(drop_mask)
        logit_pos = gradient_adjust_layer(pair_logits[0, :])
        logit_neg = pair_logits[1, :]
        y_pred = logit_pos - logit_neg
        losses = tf.maximum(1.0 - y_pred, 0)
        loss = tf.reduce_mean(losses)
    else:
        assert False

    return loss, losses, y_pred



def model_fn_ranking_w_gradient_adjust(FLAGS):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    model_config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
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

        loss, losses, y_pred = pairwise_model(features, model_config.factor, modeling_opt, pooled_output)


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
