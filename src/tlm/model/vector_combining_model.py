import tlm.training.assignment_map
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import dropout
from models.transformer.optimization_v2 import create_simple_optimizer
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.multiple_evidence import MultiEvidenceCombiner
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments, classification_metric_fn, \
    reweight_zero

def get_init_fn(config, tvars):
    assignment_fn = tlm.training.assignment_map.get_assignment_map_as_is
    assignment_map, initialized_variable_names = assignment_fn(tvars, config.init_checkpoint)
    def init_fn():
        tf.compat.v1.train.init_from_checkpoint(config.init_checkpoint, assignment_map)

    return initialized_variable_names, init_fn


def vector_combining_model(model_config, config, special_flags=[], override_prediction_fn=None):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
          tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        vectors = features["vectors"] # [batch_size, max_unit, num_hidden]
        valid_mask = features["valid_mask"]
        label_ids = features["label_ids"]
        vectors = tf.reshape(vectors, [-1, model_config.num_window, model_config.max_sequence, model_config.hidden_size])
        valid_mask = tf.reshape(valid_mask, [-1, model_config.num_window, model_config.max_sequence])
        label_ids = tf.reshape(label_ids, [-1])

        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = MultiEvidenceCombiner(
            config=model_config,
            is_training=is_training,
            vectors=vectors,
            valid_mask=valid_mask,
            scope=None
        )
        pooled = model.pooled_output
        if is_training:
            pooled = dropout(pooled, 0.1)

        logits = tf.keras.layers.Dense(config.num_classes, name="cls_dense")(pooled)
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)

        if "bias_loss" in special_flags:
            tf_logging.info("Using special_flags : bias_loss")
            loss_arr = reweight_zero(label_ids, loss_arr)

        loss = tf.reduce_mean(input_tensor=loss_arr)
        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}

        scaffold_fn = None
        if config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(config, tvars)
            scaffold_fn = get_tpu_scaffold_or_init(init_fn, config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)

        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            if "simple_optimizer" in special_flags:
                tf_logging.info("using simple optimizer")
                train_op = create_simple_optimizer(loss, config.learning_rate, config.use_tpu)
            else:
                if "ask_tvar" in special_flags:
                    tvars = model.get_trainable_vars()
                else:
                    tvars = None
                train_op = optimization.create_optimizer_from_config(loss, config, tvars)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (classification_metric_fn, [
                logits, label_ids, is_real_example
            ])
            output_spec = TPUEstimatorSpec(mode=model, loss=loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "logits": logits,
                    "label_ids": label_ids
            }
            if override_prediction_fn is not None:
                predictions = override_prediction_fn(predictions, model)

            useful_inputs = ["data_id", "input_ids2"]
            for input_name in useful_inputs:
                if input_name in features:
                    predictions[input_name] = features[input_name]
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec
    return model_fn
