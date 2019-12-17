import tensorflow as tf

from models.transformer import optimization_v2 as optimization
from tf_util.tf_logging import tf_logging
from tlm.tlm.loss_diff_common import IndependentLossModel, get_diff_loss, recover_mask, get_gold_diff
from tlm.training.assignment_map import get_bert_assignment_map
from tlm.training.input_fn_common import get_lm_basic_features, get_lm_mask_features, format_dataset


def loss_diff_prediction_model(bert_config, train_config, model_class, model_config):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        loss_base = features["loss_base"]
        loss_target = features["loss_target"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        input_ids = recover_mask(input_ids, masked_lm_positions, masked_lm_ids)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = model_class(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )

        if model_config.loss_model == "independent":
            loss_model = IndependentLossModel(bert_config)
            loss_model.train_modeling(model.get_sequence_output(), masked_lm_positions, masked_lm_weights,
                       loss_base, loss_target)

            total_loss = loss_model.total_loss
            loss1 = loss_model.loss1
            loss2 = loss_model.loss2
            per_example_loss1 = loss_model.per_example_loss1
            per_example_loss2 = loss_model.per_example_loss2
            losses1 = tf.reduce_sum(per_example_loss1, axis=1)
            losses2 = tf.reduce_sum(per_example_loss2, axis=1)
            prob1 = loss_model.prob1
            prob2 = loss_model.prob2

            def host_call_fn(total_loss, loss1, loss2):
                tf.summary.scalar("total_loss", total_loss[0])
                tf.summary.scalar("loss_base", loss1[0])
                tf.summary.scalar("loss_target", loss2[0])
                return tf.compat.v1.summary.all_v2_summary_ops()

            host_call = (host_call_fn, [tf.reshape(total_loss, [1]),
                                        tf.reshape(loss1, [1]),
                                        tf.reshape(loss2, [1])])

        elif model_config.loss_model == "diff_regression":
            total_loss, losses, logits = get_diff_loss(bert_config, model.get_sequence_output(),
                                                       masked_lm_positions, masked_lm_weights, loss_base, loss_target)
            host_call = None

        pred_diff = prob1 - prob2
        gold_diff = get_gold_diff(loss_base, loss_target)
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
                tf.compat.v1.tain.init_from_checkpoint(train_config.init_checkpoint, assignment_map)

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
                    host_call=host_call,
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss1, per_example_loss2):
                loss1 = tf.compat.v1.metrics.mean(
                    values=per_example_loss1)
                loss2 = tf.compat.v1.metrics.mean(
                    values=per_example_loss2)

                pel = per_example_loss1 + per_example_loss2

                return {
                #    "eval_loss": loss,
                    "loss1": loss1,
                    "loss2": loss2,
                }

            eval_metrics = (metric_fn, [losses1, losses2])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "loss_base": loss_base,
                    "loss_target": loss_target,
                    "prob1": prob1,
                    "prob2": prob2,
                    "per_example_loss1": per_example_loss1,
                    "per_example_loss2": per_example_loss2,
                    "input_ids":input_ids,
                    "masked_lm_positions":masked_lm_positions,
                    "pred_diff": pred_diff,
                    "gold_diff": gold_diff,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def input_fn_builder_masked(input_files, flags, is_training, num_cpu_threads=4):
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        all_features = {}
        all_features.update(get_lm_basic_features(flags))
        all_features.update(get_lm_mask_features(flags))

        active_feature = ["input_ids", "input_mask", "segment_ids",
                          "next_sentence_labels",
                          "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"
                          ]
        selected_features = {k: all_features[k] for k in active_feature}
        FixedLenFeature = tf.io.FixedLenFeature
        max_predictions_per_seq = flags.max_predictions_per_seq
        selected_features["loss_base"] = FixedLenFeature([max_predictions_per_seq], tf.float32)
        selected_features["loss_target"] = FixedLenFeature([max_predictions_per_seq], tf.float32)
        return format_dataset(selected_features, batch_size, is_training, flags, input_files, num_cpu_threads)
    return input_fn

