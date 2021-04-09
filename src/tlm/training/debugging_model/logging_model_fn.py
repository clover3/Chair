import tensorflow as tf

from models.transformer import optimization_v2 as optimization


def metric_fn(log_probs, ):

    return {}

def model_fn_logging_debug(model_config, train_config):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        output_weights = tf.compat.v1.get_variable(
            "output_weights", [10, 100],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
        )

        logits = output_weights[:, 1]
        loss = tf.reduce_mean(output_weights)
        loss2 = tf.reduce_mean(tf.square(output_weights))

        loss2_metric_val, loss2_metric_op = tf.compat.v1.metrics.mean(loss2)
        tf.compat.v1.summary.scalar("loss2", loss2)
        scaffold_fn = None
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            tvars = None
            train_op = optimization.create_optimizer_from_config(loss, train_config, tvars)
            output_spec = TPUEstimatorSpec(mode=mode,
                                           loss=loss,
                                           train_op=train_op,
                                           scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn, [
                logits
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
        else:
            predictions = {
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