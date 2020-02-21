from functools import partial

import tensorflow as tf

from data_generator.special_tokens import MASK_ID, EOW_ID, CLS_ID
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import create_initializer, get_shape_list, get_shape_list2, dropout
from models.transformer.optimization_v2 import create_optimizer_from_config
from tf_util.tf_logging import tf_logging
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.masking import random_masking
from tlm.sero.sero_core import split_and_append_sep, SeroDelta, SeroEpsilon
from tlm.training import assignment_map, grad_accumulation
from tlm.training.input_fn_common import format_dataset
from tlm.training.model_fn_common import log_features, get_tpu_scaffold_or_init, log_var_assignments, align_checkpoint, \
    Classification, reweight_zero
from tlm.training.ranking_model_common import combine_paired_input_features, get_prediction_structure, \
    apply_loss_modeling
from tlm.training.ranking_model_fn import checkpoint_init, \
    ranking_estimator_spec, rank_predict_estimator_spec
from trainer.tf_train_module_v2 import OomReportingHook


def r3to2(t):
    a, b, c = get_shape_list2(t)
    return tf.reshape(t, [-1, c])


def get_assignment_map_from_checkpoint_type(checkpoint_type, num_lower_layers):
    if checkpoint_type == "bert":
        assignment_fn = partial(assignment_map.sero_from_bert, num_lower_layers)
    elif checkpoint_type == "v2":
        assignment_fn = assignment_map.sero_from_v2
    elif checkpoint_type == "sero":
        assignment_fn = assignment_map.assignment_map_v2_to_v2
    else:
        raise Exception("Undefined checkpoint exists")

    return assignment_fn

def model_fn_sero_lm(config, train_config, modeling, prediction_op=None):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_sero_lm")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids = features["input_ids"] # [batch_size, seq_length]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        is_sero_modeling = "sero" in modeling
        if is_sero_modeling:
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
        if is_sero_modeling:
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

        if is_sero_modeling:
            masked_input_ids = tf.reshape(masked_input_ids_2d, stacked_input_ids.shape)
        elif modeling == "bert":
            masked_input_ids = tf.expand_dims(masked_input_ids_2d, 1)
            stacked_input_mask = tf.expand_dims(stacked_input_mask, 1)
            stacked_segment_ids = tf.expand_dims(stacked_segment_ids, 1)
        else:
            assert False

        if modeling == "sero":
            model_class = SeroDelta
        elif modeling == "sero_epsilon":
            model_class = SeroEpsilon

        with tf.compat.v1.variable_scope("sero"):
            model = model_class(
                config,
                is_training,
                train_config.use_one_hot_embeddings
            )
            sequence_output_3d = model.network_stacked(masked_input_ids, stacked_input_mask, stacked_segment_ids, use_context)
        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs \
            = get_masked_lm_output(config, sequence_output_3d, model.get_embedding_table(),
                                     masked_lm_positions_2d, masked_lm_ids_2d, masked_lm_weights_2d)

        predictions = None
        if prediction_op == "gradient_to_long_context":
            predictions = {}
            for idx, input_tensor in enumerate(model.upper_module_inputs):
                g = tf.abs(tf.gradients(ys=masked_lm_loss, xs=input_tensor)[0])
                main_g = g[:, :config.window_size, :]
                context_g = g[:, config.window_size:, :]
                main_g = tf.reduce_mean(tf.reduce_mean(main_g, axis=2), axis=1)
                context_g = tf.reduce_mean(tf.reduce_mean(context_g, axis=2), axis=1)
                predictions['main_g_{}'.format(idx)] = main_g
                predictions['context_g_{}'.format(idx)] = context_g


        loss = masked_lm_loss  #+ bert_task.masked_lm_loss
        tvars = tf.compat.v1.trainable_variables()
        if train_config.init_checkpoint:
            assignment_fn = get_assignment_map_from_checkpoint_type(train_config.checkpoint_type, config.lower_layers)
        else:
            assignment_fn = None
        initialized_variable_names, init_fn = align_checkpoint(tvars, train_config.init_checkpoint, assignment_fn)
        log_var_assignments(tvars, initialized_variable_names)
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)

        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer_from_config(loss, train_config)
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    training_hooks=[OomReportingHook()],
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            output_spec = TPUEstimatorSpec(mode=model, loss=loss, eval_metrics=None,
                                           scaffold_fn=scaffold_fn)
        else:
            if predictions is None:
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


def model_fn_sero_ranking_train(config, train_config, model_class):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_ranking")
        log_features(features)

        input_ids, input_mask, segment_ids = combine_paired_input_features(features)
        batch_size, _ = get_shape_list(input_mask) # This is not real batch_size, 2 * real_batch_size
        use_context = tf.ones([batch_size, 1], tf.int32)


        stacked_input_ids, stacked_input_mask, stacked_segment_ids, \
            = split_and_append_sep(input_ids, input_mask, segment_ids,
                                   config.total_sequence_length, config.window_size, CLS_ID, EOW_ID)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        with tf.compat.v1.variable_scope("sero"):
            model = model_class(
                config,
                is_training,
                train_config.use_one_hot_embeddings
            )
            sequence_output_3d = model.network_stacked(stacked_input_ids, stacked_input_mask,
                                                       stacked_segment_ids, use_context)

        pooled_output = model.get_pooled_output()

        if is_training:
            pooled_output = dropout(pooled_output, 0.1)

        loss, losses, y_pred = apply_loss_modeling(config.loss, pooled_output, features)

        assignment_fn = get_assignment_map_from_checkpoint_type(train_config.checkpoint_type, config.lower_layers)
        scaffold_fn = checkpoint_init(assignment_fn, train_config)
        prediction = {
            "stacked_input_ids": stacked_input_ids,
            "stacked_input_mask": stacked_input_mask,
            "stacked_segment_ids": stacked_segment_ids,
        }

        if train_config.gradient_accumulation != 1:
            optimizer_factory = lambda x: grad_accumulation.get_accumulated_optimizer_from_config(x, train_config,
                                                  tf.compat.v1.trainable_variables(), train_config.gradient_accumulation)
        else:
            optimizer_factory = lambda x: create_optimizer_from_config(x, train_config)
        return ranking_estimator_spec(mode, loss, losses, y_pred, scaffold_fn, optimizer_factory, prediction)

    return model_fn


def model_fn_sero_ranking_predict(config, train_config, model_class):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_sero_ranking_predict")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        batch_size, _ = get_shape_list(input_mask)
        use_context = tf.ones([batch_size, 1], tf.int32)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids, \
            = split_and_append_sep(input_ids, input_mask, segment_ids,
                                   config.total_sequence_length, config.window_size, CLS_ID, EOW_ID)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # Updated

        with tf.compat.v1.variable_scope("sero"):
            model = model_class(
                config,
                is_training,
                train_config.use_one_hot_embeddings
            )
            model.network_stacked(stacked_input_ids, stacked_input_mask, stacked_segment_ids, use_context)

        pooled_output = model.get_pooled_output()
        logits = get_prediction_structure(config.loss, pooled_output)

        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = assignment_map.assignment_map_v2_to_v2

        initialized_variable_names, init_fn = align_checkpoint(tvars, train_config.init_checkpoint, assignment_fn)
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)
        output_spec = rank_predict_estimator_spec(logits, mode, scaffold_fn)
        return output_spec


    return model_fn


def model_fn_sero_classification(config, train_config, modeling, special_flags=[]):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_sero_classification")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        batch_size, _ = get_shape_list(input_mask)
        use_context = tf.ones([batch_size, 1], tf.int32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # Updated
        if modeling == "sero":
            model_class = SeroDelta
            print("Using SeroDelta")
        elif modeling == "sero_epsilon":
            model_class = SeroEpsilon
            print("Using SeroEpsilon")
        else:
            assert False

        with tf.compat.v1.variable_scope("sero"):
            model = model_class(
                config,
                is_training,
                train_config.use_one_hot_embeddings
            )
            input_ids = tf.expand_dims(input_ids, 1)
            input_mask = tf.expand_dims(input_mask, 1)
            segment_ids = tf.expand_dims(segment_ids, 1)
            sequence_output = model.network_stacked(input_ids, input_mask, segment_ids, use_context)

        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = tf.keras.layers.Dense(config.hidden_size,
                                               activation=tf.keras.activations.tanh,
                                               kernel_initializer=create_initializer(config.initializer_range))(
            first_token_tensor)

        if "bias_loss" in special_flags:
            loss_weighting = reweight_zero
        else:
            loss_weighting = None

        task = Classification(3, features, pooled_output, is_training, loss_weighting)
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
            predictions = {
                "input_ids": input_ids,
                "logits": task.logits
            }
            output_spec = TPUEstimatorSpec(mode=model, loss=loss, predictions=predictions,
                                           scaffold_fn=scaffold_fn)
        return output_spec
    return model_fn


def input_fn_builder(input_files, total_sequence_length, flags, is_training, num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        FixedLenFeature = tf.io.FixedLenFeature
        all_features = {
            "input_ids":    FixedLenFeature([total_sequence_length], tf.int64),
            "input_mask":   FixedLenFeature([total_sequence_length], tf.int64),
            "segment_ids":  FixedLenFeature([total_sequence_length], tf.int64),
            "use_context":  FixedLenFeature([1], tf.int64),
        }
        return format_dataset(all_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn
