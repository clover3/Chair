import tensorflow as tf

from models.transformer import optimization_v2 as optimization
from tf_util.tf_logging import tf_logging
from tlm.training.assignment_map import get_bert_assignment_map


def token_scoring_model(bert_config, train_config, model_class, model_config):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        label_masks = features["label_masks"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = model_class(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )
        logits = tf.keras.layers.Dense(train_config.num_classes, name="token_regression")(model.get_sequence_output())

        per_ex_losses = tf.keras.losses.MAE(tf.expand_dims(label_ids, 2), logits)
        masked_losses = per_ex_losses * tf.cast(label_masks, tf.float32)
        losses_sum = tf.reduce_sum(masked_losses, axis=1)
        denom = tf.reduce_sum(tf.cast(label_masks, tf.float32), axis=1) + 1e-5
        losses = losses_sum / denom
        total_loss = tf.reduce_mean(losses)
        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if train_config.init_checkpoint:
            assignment_map, initialized_variable_names = get_bert_assignment_map(tvars, train_config.init_checkpoint)
            if train_config.use_tpu:
                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)

        tf_logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf_logging.info("name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer_from_config(total_loss, train_config)
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(logits, label_ids, label_masks):
                logits_reduced = tf.squeeze(logits, 2)
                is_neg_correct = tf.logical_and(tf.less(label_ids, 0.), tf.less(logits_reduced, 0.))
                is_pos_correct = tf.logical_and(tf.less(0., label_ids), tf.less(0., logits_reduced))
                is_correct = tf.logical_or(is_neg_correct, is_pos_correct)

                float_masks = tf.cast(label_masks, tf.float32)
                num_correct = tf.reduce_sum(tf.cast(is_correct, tf.float32) * float_masks, axis=1)
                num_problems = tf.reduce_sum(float_masks, axis=1) + 1e-5
                acc_list = num_correct / num_problems

                mean_acc = tf.compat.v1.metrics.mean(
                    values=acc_list)

                return {
                    'mean_acc': mean_acc
                }

            eval_metrics = (metric_fn, [logits, label_ids, label_masks])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "logits": logits,
                    "input_ids":input_ids,
                    "labels": label_ids,
                    "label_masks": label_masks,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn

