
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import dropout
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.training.classification_model_fn import get_init_fn
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments, classification_metric_fn




def model_fn_classification_weighted_loss(model_config, train_config):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        label_ids = tf.reshape(label_ids, [-1])
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model_1 = BertModel(
            config=model_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )
        pooled = model_1.get_pooled_output()
        if is_training:
            pooled = dropout(pooled, 0.1)
        logits = tf.keras.layers.Dense(train_config.num_classes, name="cls_dense")(pooled)

        logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)
        y_pred = tf.nn.softmax(logits, axis=-1)
        eps = 1e-9
        y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
        alpha = 0.1
        beta = 0.1
        y_true = tf.one_hot(label_ids, train_config.num_classes)
        y_true = tf.cast(y_true, tf.float32)
        T = tf.constant([[1-alpha, alpha],
                         [alpha, 1-alpha]])
        y_pred_p = tf.matmul(y_pred, T)
        loss_arr = tf.reduce_sum(-y_true * tf.math.log(y_pred_p), axis=1)
        loss = tf.reduce_mean(input_tensor=loss_arr)
        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, tvars)
            scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            tvars = None
            train_op = optimization.create_optimizer_from_config(loss, train_config, tvars)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (classification_metric_fn, [
                logits, label_ids, is_real_example
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "input_ids": input_ids,
                    "label_ids": label_ids,
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
