from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import dropout
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments, classification_metric_fn


def layernorm_dot_product(logits, label_ids):
    y_pred = logits
    y_true = (tf.cast(label_ids, tf.float32) - 0.5) * 2

    y_pred_std = tf.keras.layers.BatchNormalization()(y_pred)
    r = tf.multiply(y_pred_std, y_true)
    return -r


def forward(logits, label_ids, param):
    alpha = param['alpha']
    beta = param['beta']
    probabilities = tf.nn.softmax(logits, axis=1)
    t_map = [[1. - beta, 0. + alpha],
             [0. + beta, 1. - alpha]]
    adjusted_probabilies = tf.matmul(probabilities, t_map)
    epsilon = 1e-6
    y_pred = tf.clip_by_value(adjusted_probabilies, epsilon, 1.0 - epsilon)
    y_true = tf.one_hot(label_ids, 2)
    losses = -tf.reduce_sum(y_true * tf.math.log(y_pred), 1)
    return losses


def generalized_cross_entropy(logits, label_ids):
    """
    2018 - nips - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels.
    """
    y_pred = tf.nn.softmax(logits, axis=1)
    y_true = tf.one_hot(label_ids, 2)
    q = 0.7
    t_loss = (1 - tf.pow(tf.reduce_sum(y_true * y_pred, axis=-1), q)) / q
    return tf.reduce_mean(t_loss)


def apply_loss(logits, label_ids, param):
    if param['loss_type'] == "layernorm_dot_product":
        losses = layernorm_dot_product(logits, label_ids)
    elif param['loss_type'] == "forward":
        losses = forward(logits, label_ids, param)
    elif param['loss_type'] == "generalized_cross_entropy":
        losses = generalized_cross_entropy(logits, label_ids)
    else:
        assert False

    loss = tf.reduce_mean(losses)
    return loss


def model_fn_classification_with_alt_loss(model_config, train_config, model_class, special_flags=[], override_prediction_fn=None):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        if mode == tf.estimator.ModeKeys.PREDICT:
            label_ids = tf.ones([input_ids.shape[0]], dtype=tf.int32)
        else:
            label_ids = features["label_ids"]
            label_ids = tf.reshape(label_ids, [-1])
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if "feed_features" in special_flags:
            model = model_class(
                config=model_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
                features=features,
            )
        else:
            model = model_class(
                config=model_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )
        pooled = model.get_pooled_output()
        if is_training:
            pooled = dropout(pooled, 0.1)

        logits = tf.keras.layers.Dense(train_config.num_classes, name="cls_dense")(pooled)

        loss = apply_loss(logits, label_ids, model_config.to_dict())
        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, tvars)
            scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec

        def metric_fn(log_probs, label, is_real_example, confidence):
            r = classification_metric_fn(log_probs, label, is_real_example)
            r['confidence'] = tf.compat.v1.metrics.mean(confidence)
            return r

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            tvars = None
            train_op = optimization.create_optimizer_from_config(loss, train_config, tvars)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn, [
                logits, label_ids, is_real_example, confidence
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "input_ids": input_ids,
                    "logits": logits,
                    "confidence": confidence,
            }
            if "data_id" in features:
                predictions['data_id'] = features['data_id']
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)
        return output_spec
    return model_fn
