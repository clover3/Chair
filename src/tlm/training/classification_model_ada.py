
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import dropout
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.training.classification_model_fn import get_init_fn
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments, classification_metric_fn

@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad


def model_fn_classification_with_ada(model_config, train_config):
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

        domain_ids = features["domain_ids"]
        domain_ids = tf.reshape(domain_ids, [-1])

        is_valid_label = features["is_valid_label"]

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
        pred_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)
        num_domain = 2
        pooled_for_domain = grad_reverse(pooled)
        domain_logits = tf.keras.layers.Dense(num_domain, name="domain_dense")(pooled_for_domain)
        domain_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=domain_logits,
            labels=domain_ids)

        pred_loss = tf.reduce_mean(pred_losses * tf.cast(is_valid_label, tf.float32))
        domain_loss = tf.reduce_mean(domain_losses)

        tf.compat.v1.summary.scalar('domain_loss', domain_loss)
        tf.compat.v1.summary.scalar('pred_loss', pred_loss)
        alpha = model_config.alpha
        loss = pred_loss + alpha * domain_loss
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
