from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import dropout
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments


def model_fn_regression(model_config, train_config):
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
            label_ids = tf.ones([input_ids.shape[0]], dtype=tf.float32)
        else:
            label_ids = features["label_ids"]
            label_ids = tf.reshape(label_ids, [-1])
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = BertModel(
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
        scale = model_config.scale

        label_ids = scale * label_ids

        weight = tf.abs(label_ids)
        loss_arr = tf.keras.losses.MAE(y_true=label_ids, y_pred=logits)
        loss_arr = loss_arr * weight

        loss = tf.reduce_mean(input_tensor=loss_arr)
        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None

        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, tvars)
            scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec

        def metric_fn(logits, label, is_real_example):
            mae = tf.compat.v1.metrics.mean_absolute_error(
                labels=label, predictions=logits, weights=is_real_example)

            return {
                "mae": mae
            }

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            tvars = None
            train_op = optimization.create_optimizer_from_config(loss, train_config, tvars)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn, [
                logits, label_ids, is_real_example
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "input_ids": input_ids,
                    "logits": logits,
            }
            if "data_id" in features:
                predictions['data_id'] = features['data_id']
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)
        return output_spec
    return model_fn
