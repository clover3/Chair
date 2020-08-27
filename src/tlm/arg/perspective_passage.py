import tensorflow as tf

from models.transformer import optimization_v2 as optimization
from tf_util.tf_logging import tf_logging
from tlm.training.assignment_map import get_bert_assignment_map
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn_common import format_dataset
from tlm.training.ranking_model_common import combine_paired_input_features


def input_fn_perspective_passage(flags):
    input_files = get_input_files_from_flags(flags)
    max_seq_length = flags.max_seq_length
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = dict({
                "input_ids1": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask1": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids1": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "input_mask2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
                "segment_ids2": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        })
        name_to_features["strict_good"] = tf.io.FixedLenFeature([1], tf.int64)
        name_to_features["strict_bad"] = tf.io.FixedLenFeature([1], tf.int64)
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn

def cast_float_multiply(a, b):
    return tf.cast(a, tf.float32) * tf.cast(b, tf.float32)


def pairwise_model(pooled_output, strict_good, strict_bad):
    logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled_output)
    pair_logits = tf.reshape(logits, [2, -1])
    y_pred = pair_logits[0, :] - pair_logits[1, :]

    hinge_losses = tf.maximum(1.0 - y_pred, 0)
    raw_good_losses = tf.maximum(1.0 - pair_logits[0, :], 0)
    raw_bad_losses = tf.maximum(1.0 + pair_logits[1, :], 0)

    good_losses = raw_good_losses * tf.cast(strict_good, tf.float32)
    bad_losses = raw_bad_losses * tf.cast(strict_bad, tf.float32)

    losses = hinge_losses + good_losses + bad_losses
    return losses, logits, pair_logits


def ppnc_pairwise_model(bert_config, train_config, model_class, model_config):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("name = %s, shape = %s" % (name, features[name].shape))
        input_ids, input_mask, segment_ids = combine_paired_input_features(features)

        strict_good = features["strict_good"]
        strict_bad = features["strict_bad"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = model_class(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )
        pooled = model.get_pooled_output()
        losses, logits, pair_logits = pairwise_model(pooled, strict_good, strict_bad)
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
            def metric_fn(pair_logits, strict_good, strict_bad):
                diff = pair_logits[0, :] - pair_logits[1, :]
                pairwise_correct = tf.less(0.0, diff)

                strict_good_correct_raw = tf.reshape(tf.less(1.0, pair_logits[0, :]), [-1, 1])
                strict_good_correct = cast_float_multiply(strict_good_correct_raw, strict_good)
                strict_bad_correct_raw = tf.reshape(tf.less(pair_logits[1, :], -1.0), [-1, 1])
                strict_bad_correct = cast_float_multiply(strict_bad_correct_raw, strict_bad)

                pairwise_acc_raw = tf.cast(pairwise_correct, tf.float32)
                mean_acc = tf.compat.v1.metrics.mean(values=pairwise_acc_raw)

                def strict_accuracy(correctness, gold):
                    return tf.compat.v1.metrics.accuracy(labels=tf.ones_like(gold, tf.int32),
                                                         predictions=tf.cast(correctness, tf.int32),
                                                         weights=tf.cast(gold, tf.float32))

                return {
                    'mean_acc': mean_acc,
                    'strict_good_acc': strict_accuracy(strict_good_correct, strict_good),
                    'strict_bad_acc': strict_accuracy(strict_bad_correct, strict_bad)
                }

            eval_metrics = (metric_fn, [pair_logits, strict_good, strict_bad])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "logits": logits,
                    "input_ids": input_ids,
                    "strict_good": strict_good,
                    "strict_bad": strict_bad,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn

