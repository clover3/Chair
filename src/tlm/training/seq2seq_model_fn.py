import tensorflow as tf

import tlm.training.assignment_map
from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.masking import random_masking
from tlm.training.input_fn_common import format_dataset
from tlm.training.lm_model_fn import metric_fn_lm
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments, log_features, align_checkpoint


def mask_lm_as_seq2seq(config, train_config):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_apr_lm")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        raw_input_ids = features["input_ids"] # [batch_size, seq_length]
        raw_input_mask = features["input_mask"]
        raw_segment_ids = features["segment_ids"]

        word_tokens = features["word"]
        word_input_mask = tf.cast(tf.not_equal(word_tokens, 0), tf.int32)
        word_segment_ids = tf.ones_like(word_tokens, tf.int32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.random.set_seed(0)
            seed = 0
        else:
            seed = None

        input_ids = tf.concat([word_tokens, raw_input_ids], axis=1)
        input_mask = tf.concat([word_input_mask, raw_input_mask], axis=1)
        segment_ids = tf.concat([word_segment_ids, raw_segment_ids], axis=1)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        tf_logging.info("Using masked_input_ids")
        masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
            = random_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID, seed)

        model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=masked_input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
                 config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)

        loss = masked_lm_loss
        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = tlm.training.assignment_map.get_bert_assignment_map
        initialized_variable_names, init_fn = align_checkpoint(tvars, train_config.init_checkpoint, assignment_fn)
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)

        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_logging.info("Using single lr ")
            train_op = optimization.create_optimizer_from_config(loss, train_config)
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn_lm, [
                    masked_lm_example_loss,
                    masked_lm_log_probs,
                    masked_lm_ids,
                    masked_lm_weights,
            ])
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "input_ids": input_ids,
                    "masked_input_ids": masked_input_ids,
                    "masked_lm_ids": masked_lm_ids,
                    "masked_lm_example_loss": masked_lm_example_loss,
                    "masked_lm_positions": masked_lm_positions
            }
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec
    return model_fn


def input_fn_builder(input_files, flags, is_training, num_cpu_threads=4):
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_sequence_length = flags.max_seq_length
        FixedLenFeature = tf.io.FixedLenFeature
        all_features = {
            "input_ids":    FixedLenFeature([max_sequence_length], tf.int64),
            "input_mask":   FixedLenFeature([max_sequence_length], tf.int64),
            "segment_ids":  FixedLenFeature([max_sequence_length], tf.int64),
            "word": FixedLenFeature([flags.max_word_length], tf.int64),
        }
        return format_dataset(all_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn
