import abc
from collections import Callable

from models.transformer import optimization_v2 as optimization
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments


def metric_fn(log_probs, ):
    return {}


class ModelLight(abc.ABC):
    @abc.abstractmethod
    def get_logits(self):
        pass

    @abc.abstractmethod
    def get_loss(self):
        pass


def model_fn_with_loss(model_config, train_config, model_class: Callable[ModelLight]):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model: ModelLight = model_class(
            config=model_config,
            is_training=is_training,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            features=features,
        )
        logits = model.get_logits()
        loss = model.get_loss()
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