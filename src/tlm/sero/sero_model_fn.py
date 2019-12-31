import tensorflow as tf

from data_generator.special_tokens import MASK_ID, EOW_ID, CLS_ID
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import create_initializer, get_shape_list, get_shape_list2
from tf_util.tf_logging import tf_logging
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.masking import random_masking
from tlm.sero.sero_core import SeroAlpha, split_and_append_sep, SeroGamma
from tlm.training import assignment_map
from tlm.training.input_fn_common import format_dataset
from tlm.training.lm_model_fn import metric_fn_lm
from tlm.training.model_fn_common import log_features, get_tpu_scaffold_or_init, log_var_assignments, align_checkpoint, \
    Classification
from trainer.tf_train_module_v2 import OomReportingHook


def r3to2(t):
    a, b, c = get_shape_list2(t)
    return tf.reshape(t, [-1, c])


def model_fn_sero_lm(config, train_config, modeling):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_apr_lm")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids = features["input_ids"] # [batch_size, seq_length]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        if modeling == "sero":
            use_context = features["use_context"]
        elif modeling == "bert":
            batch_size, _ = get_shape_list(input_mask)
            use_context = tf.ones([batch_size, 1], tf.int32)
        else:
            assert False

        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.random.set_seed(0)
            seed = 0
        else:
            seed = None

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        tf_logging.info("Using masked_input_ids")
        if modeling == "sero":
            stacked_input_ids, stacked_input_mask, stacked_segment_ids, \
                = split_and_append_sep(input_ids, input_mask, segment_ids,
                                       config.total_sequence_length, config.window_size, CLS_ID, EOW_ID)
            input_ids_2d = r3to2(stacked_input_ids)
            input_mask_2d = r3to2(stacked_input_mask)

        elif modeling == "bert":
            stacked_input_ids, stacked_input_mask, stacked_segment_ids = input_ids, input_mask, segment_ids
            input_ids_2d = stacked_input_ids
            input_mask_2d = stacked_input_mask
        else:
            assert False

        tf_logging.info("Doing dynamic masking (random)")

        # TODO make stacked_input_ids 2D and recover
        masked_input_ids_2d, masked_lm_positions_2d, masked_lm_ids_2d, masked_lm_weights_2d \
            = random_masking(input_ids_2d, input_mask_2d,
                             train_config.max_predictions_per_seq, MASK_ID, seed, [EOW_ID])

        if modeling == "sero":
            masked_input_ids = tf.reshape(masked_input_ids_2d, stacked_input_ids.shape)
        elif modeling == "bert":
            masked_input_ids = tf.expand_dims(masked_input_ids_2d, 1)
            stacked_input_mask = tf.expand_dims(stacked_input_mask, 1)
            stacked_segment_ids = tf.expand_dims(stacked_segment_ids, 1)
        else:
            assert False

        model_class = SeroGamma

        with tf.compat.v1.variable_scope("sero"):
            model = model_class(
                config,
                is_training,
                train_config.use_one_hot_embeddings
            )
            sequence_output_3d = model.call(masked_input_ids, stacked_input_mask, stacked_segment_ids, use_context)
        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs \
            = get_masked_lm_output(config, sequence_output_3d, model.get_embedding_table(),
                                     masked_lm_positions_2d, masked_lm_ids_2d, masked_lm_weights_2d)

        loss = masked_lm_loss  #+ bert_task.masked_lm_loss
        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = assignment_map.sero_from_bert
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
                    training_hooks=[OomReportingHook()],
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn_lm, [
                    masked_lm_example_loss,
                    masked_lm_log_probs,
                    masked_lm_ids_2d,
                    masked_lm_weights_2d,
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
                    "masked_lm_ids": masked_lm_ids_2d,
                    "masked_lm_example_loss": masked_lm_example_loss,
                    "masked_lm_positions": masked_lm_positions_2d,
            }
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec
    return model_fn



def model_fn_sero_classification(config, train_config, modeling):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_apr_lm")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        batch_size, _ = get_shape_list(input_mask)
        use_context = tf.ones([batch_size, 1], tf.int32)

        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.random.set_seed(0)
            seed = 0
        else:
            seed = None

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        tf_logging.info("Doing dynamic masking (random)")

        with tf.compat.v1.variable_scope("sero"):
            model = SeroAlpha(
                config,
                is_training,
                train_config.use_one_hot_embeddings
            )
            sequence_output = model.call(input_ids, input_mask, segment_ids, use_context)

        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = tf.keras.layers.Dense(config.hidden_size,
                                               activation=tf.keras.activations.tanh,
                                               kernel_initializer=create_initializer(config.initializer_range))(
            first_token_tensor)
        task = Classification(3, features, pooled_output, is_training)
        loss = task.loss

        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = assignment_map.assignment_map_v2_to_v2

        initialized_variable_names, init_fn = align_checkpoint(tvars, train_config.init_checkpoint, assignment_fn)
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)

        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_logging.info("Using single lr ")
            train_op = optimization.create_optimizer_from_config(loss, train_config)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            output_spec = TPUEstimatorSpec(mode=model, loss=loss, eval_metrics=task.eval_metrics(),
                                           scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = TPUEstimatorSpec(mode=model, loss=loss, predictions={"loss": task.loss_arr},
                                           scaffold_fn=scaffold_fn)
        return output_spec
    return model_fn

def input_fn_builder(input_files, flags, config, is_training, num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        total_sequence_length = config.total_sequence_length
        FixedLenFeature = tf.io.FixedLenFeature
        all_features = {
            "input_ids":    FixedLenFeature([total_sequence_length], tf.int64),
            "input_mask":   FixedLenFeature([total_sequence_length], tf.int64),
            "segment_ids":  FixedLenFeature([total_sequence_length], tf.int64),
            "use_context":  FixedLenFeature([1], tf.int64),
        }
        return format_dataset(all_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn
