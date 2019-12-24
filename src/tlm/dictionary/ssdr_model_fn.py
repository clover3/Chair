import tensorflow as tf

import tlm.training.dict_model_fn as dict_model_fn
import tlm.training.input_fn
import tlm.training.input_fn_common
from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from models.transformer.optimization_v2 import create_optimizer
from tf_util.tf_logging import tf_logging
from tlm.dictionary.sense_selecting_dictionary_reader import APR
from tlm.model.base import BertModel
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.masking import random_masking
from tlm.training.assignment_map import get_assignment_map_as_is
from tlm.training.grad_accumulation import get_accumulated_optimizer_from_config
from tlm.training.lm_model_fn import metric_fn_lm
from tlm.training.model_fn_common import log_var_assignments, get_tpu_scaffold_or_init, align_checkpoint, \
    Classification, log_features, align_checkpoint_twice
from trainer.get_param_num import get_param_num
from trainer.tf_module import split_tvars
# APR : Auxiliary Parameter Reader
from trainer.tf_train_module_v2 import OomReportingHook


def create_optimizer_with_separate_lr(loss, train_config, tvars=None):
    if tvars is None:
        tvars = tf.compat.v1.trainable_variables()
    vars1, vars2 = split_tvars(tvars, "dict")
    train_op1 = create_optimizer(
        loss,
        train_config.learning_rate,
        train_config.num_train_steps,
        train_config.num_warmup_steps,
        train_config.use_tpu,
        vars1
    )

    train_op2 = create_optimizer(
        loss,
        train_config.learning_rate2,
        train_config.num_train_steps,
        train_config.num_warmup_steps,
        train_config.use_tpu,
        vars2
    )
    train_op = tf.group([train_op1, train_op2])
    return train_op


def model_fn_apr_classification(bert_config, ssdr_config, train_config, dict_run_config):
    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        #
        #
        # d_input_ids, d_input_mask, d_segment_ids, d_location_ids, ab_mapping, ab_mapping_mask \
        #     = get_dummy_apr_input(input_ids, input_mask,
        #                           dict_run_config.def_per_batch,
        #                           dict_run_config.inner_batch_size,
        #                           ssdr_config.max_loc_length,
        #                           dict_run_config.max_def_length)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = APR(input_ids, input_mask, segment_ids, is_training, train_config.use_one_hot_embeddings,
                    bert_config, ssdr_config,
                    dict_run_config.def_per_batch,
                    dict_run_config.inner_batch_size,
                    dict_run_config.max_def_length)

        #
        # model = model_class(
        #         config=bert_config,
        #         ssdr_config=ssdr_config,
        #         is_training=is_training,
        #         input_ids=input_ids,
        #         input_mask=input_mask,
        #         token_type_ids=segment_ids,
        #         d_input_ids=d_input_ids,
        #         d_input_mask=d_input_mask,
        #         d_segment_ids=d_segment_ids,
        #         d_location_ids=d_location_ids,
        #         ab_mapping=ab_mapping,
        #         ab_mapping_mask=ab_mapping_mask,
        #         use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        # )
        task = Classification(3, features, model.get_pooled_output(), is_training)
        loss = task.loss

        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = tlm.training.assignment_map.get_assignment_map_as_is
        initialized_variable_names, init_fn = align_checkpoint(tvars, train_config.init_checkpoint, assignment_fn)
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)
        output_spec = None
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            if ssdr_config.compare_attrib_value_safe("use_two_lr", True):
                tf_logging.info("Using two lr for each parts")
                train_op = create_optimizer_with_separate_lr(loss, train_config)
            else:
                tf_logging.info("Using single lr ")
                train_op = optimization.create_optimizer_from_config(loss, train_config)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            output_spec = TPUEstimatorSpec(mode=model, loss=loss, eval_metrics=task.eval_metrics(),
                                           scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = TPUEstimatorSpec(mode=model, loss=loss, predictions={"loss":task.loss_arr},
                                           scaffold_fn=scaffold_fn)
        return output_spec
    return model_fn


def print_all_tensor():
    return
    graph = tf.compat.v1.get_default_graph()
    tensors_per_node = [op.values() for op in graph.get_operations()]
    for tensors in tensors_per_node:
        for t in tensors:
            print(t.name, t.shape)


def model_fn_apr_lm(bert_config, ssdr_config, train_config, dict_run_config):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_apr_lm")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.random.set_seed(0)
            seed = 0
        else:
            seed = None

        tf_logging.info("Doing dynamic masking (random)")
        masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
            = random_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID, seed)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        tf_logging.info("Using masked_input_ids")
        model = APR(masked_input_ids,
                    input_mask,
                    segment_ids,
                    is_training,
                    train_config.use_one_hot_embeddings,
                    bert_config,
                    ssdr_config,
                    dict_run_config.def_per_batch,
                    dict_run_config.inner_batch_size,
                    dict_run_config.max_def_length,
                   #  MainTransformer,
                   #  SecondTransformerEmbeddingLess,
                     )

        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs \
            = get_masked_lm_output(bert_config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)

        loss = masked_lm_loss

        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = dict_model_fn.get_bert_assignment_map_for_dict
        initialized_variable_names, init_fn = align_checkpoint_twice(tvars, train_config.init_checkpoint, assignment_fn)
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)

        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            if ssdr_config.compare_attrib_value_safe("use_two_lr", True):
                tf_logging.info("Using two lr for each parts")
                train_op = create_optimizer_with_separate_lr(loss, train_config)
            else:
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
                    masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
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
                    "masked_lm_positions": masked_lm_positions,
            }
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec
    return model_fn


def model_fn_dict_reader(bert_config, ssdr_config, train_config, logging, model_class, dict_run_config):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        def reform_a_input(raw_input):
            return tf.reshape(raw_input, [dict_run_config.inner_batch_size, -1])

        def reform_b_input(raw_input):
            return tf.reshape(raw_input, [dict_run_config.def_per_batch, -1])

        input_ids = reform_a_input(features["input_ids"])
        input_mask = reform_a_input(features["input_mask"])
        segment_ids = reform_a_input(features["segment_ids"])
        d_input_ids = reform_b_input(features["d_input_ids"])
        d_input_mask = reform_b_input(features["d_input_mask"])
        d_location_ids = reform_a_input(features["d_location_ids"])
        ab_mapping = features["ab_mapping"]

        if hasattr(ssdr_config, "blind_dictionary") and ssdr_config.blind_dictionary:
            logging.info("Hide dictionary")
            d_input_ids = tf.zeros_like(d_input_ids)
            d_input_mask = tf.zeros_like(d_input_mask)

        if dict_run_config.prediction_op == "loss":
            seed = 0
        else:
            seed = None

        if dict_run_config.prediction_op == "loss_fixed_mask" or train_config.fixed_mask:
            masked_input_ids = input_ids
            masked_lm_positions = reform_a_input(features["masked_lm_positions"])
            masked_lm_ids = reform_a_input(features["masked_lm_ids"])
            masked_lm_weights = reform_a_input(features["masked_lm_weights"])
        else:
            masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
                = random_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID, seed)

        if dict_run_config.use_d_segment_ids:
            d_segment_ids = reform_b_input(features["d_segment_ids"])
        else:
            d_segment_ids = None

        if dict_run_config.use_ab_mapping_mask:
            ab_mapping_mask = reform_a_input(features["ab_mapping_mask"])
        else:
            ab_mapping_mask = None

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = model_class(
                config=bert_config,
                ssdr_config=ssdr_config,
                is_training=is_training,
                input_ids=masked_input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                d_input_ids=d_input_ids,
                d_input_mask=d_input_mask,
                d_segment_ids=d_segment_ids,
                d_location_ids=d_location_ids,
                ab_mapping=ab_mapping,
                ab_mapping_mask=ab_mapping_mask,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )


        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
                 bert_config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)

        total_loss = masked_lm_loss
        print_all_tensor()
        tvars = tf.compat.v1.trainable_variables()

        init_vars = {}
        scaffold_fn = None
        if train_config.init_checkpoint:
            if dict_run_config.is_bert_checkpoint:
                map1, map2, init_vars = dict_model_fn.get_bert_assignment_map_for_dict(tvars, train_config.init_checkpoint)

                def load_fn():
                    tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, map1)
                    tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, map2)
            else:
                map1, init_vars = get_assignment_map_as_is(tvars, train_config.init_checkpoint)

                def load_fn():
                    tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, map1)

            if train_config.use_tpu:
                def tpu_scaffold():
                    load_fn()
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                load_fn()

        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in init_vars:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("    name = %s, shape = %s%s", var.name, var.shape, init_string)
        logging.info("Total parameters : %d" % get_param_num())

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            if train_config.gradient_accumulation == 1:
                train_op = optimization.create_optimizer_from_config(total_loss, train_config)
            else:
                logging.info("Using gradient accumulation : %d" % train_config.gradient_accumulation)
                train_op = get_accumulated_optimizer_from_config(total_loss, train_config,
                                                                 tvars, train_config.gradient_accumulation)
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn_lm, [
                    masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights
            ])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            if dict_run_config.prediction_op == "gradient":
                logging.info("Fetching gradient")
                gradient = dict_model_fn.get_gradients(model, masked_lm_log_probs,
                                                       train_config.max_predictions_per_seq, bert_config.vocab_size)
                predictions = {
                    "masked_input_ids": masked_input_ids,
                    "d_input_ids": d_input_ids,
                    "masked_lm_positions": masked_lm_positions,
                    "gradients": gradient,
                }
            elif dict_run_config.prediction_op == "scores":
                logging.info("Fetching input/d_input and scores")
                predictions = {
                    "masked_input_ids": masked_input_ids,
                    "d_input_ids": d_input_ids,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_ids": masked_lm_ids,
                    "ab_mapping":ab_mapping,
                    "d_location_ids":d_location_ids,
                    "scores": model.scores,
                }
            elif dict_run_config.prediction_op == "loss" or dict_run_config.prediction_op == "loss_fixed_mask":
                logging.info("Fetching loss")
                predictions = {
                    "masked_lm_example_loss": masked_lm_example_loss,
                }
            else:
                raise Exception("prediction target not specified")

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def model_fn_apr_debug(bert_config, ssdr_config, train_config, logging, model_name, dict_run_config):
    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        log_features(features)


        def reform_a_input(raw_input):
            return tf.reshape(raw_input, [dict_run_config.inner_batch_size, -1])

        def reform_b_input(raw_input):
            return tf.reshape(raw_input, [dict_run_config.def_per_batch, -1])

        input_ids = reform_a_input(features["input_ids"])
        input_mask = reform_a_input(features["input_mask"])
        segment_ids = reform_a_input(features["segment_ids"])
        tf_logging.info("input_ids, input_mask")

        # input_ids = features["input_ids"]
        # input_mask = features["input_mask"]
        # segment_ids = features["segment_ids"]

        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.random.set_seed(0)
            seed = 0
        else:
            seed = None

        # tf_logging.info("Doing dynamic masking (random)")
        # masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
        #     = random_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID, seed)
        # if dict_run_config.prediction_op == "loss_fixed_mask" or train_config.fixed_mask:
        masked_input_ids = input_ids
        masked_lm_positions = reform_a_input(features["masked_lm_positions"])
        masked_lm_ids = reform_a_input(features["masked_lm_ids"])
        masked_lm_weights = reform_a_input(features["masked_lm_weights"])

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if model_name == "APR":
            model = APR(
                    masked_input_ids,
                    input_mask,
                    segment_ids,
                    is_training,
                    train_config.use_one_hot_embeddings,
                    bert_config,
                    ssdr_config,
                    dict_run_config.def_per_batch,
                    dict_run_config.inner_batch_size,
                    dict_run_config.max_def_length,
            )
        elif model_name == "BERT":
            model = BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=masked_input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )
        else:
            assert False

        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs \
            = get_masked_lm_output(bert_config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)

        loss = masked_lm_loss

        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = dict_model_fn.get_bert_assignment_map_for_dict
        initialized_variable_names, init_fn = align_checkpoint_twice(tvars, train_config.init_checkpoint, assignment_fn)
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)

        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            if ssdr_config.compare_attrib_value_safe("use_two_lr", True):
                tf_logging.info("Using two lr for each parts")
                train_op = create_optimizer_with_separate_lr(loss, train_config)
            else:
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
                    masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
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
                    "masked_lm_positions": masked_lm_positions,
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
        inner_batch_size = flags.inner_batch_size
        def_per_batch = flags.def_per_batch

        max_seq_length = flags.max_seq_length * inner_batch_size
        max_predictions_per_seq = flags.max_predictions_per_seq * inner_batch_size
        max_loc_length = flags.max_loc_length * inner_batch_size
        max_word_length = flags.max_word_length * inner_batch_size

        max_def_length = flags.max_def_length * def_per_batch
        FixedLenFeature = tf.io.FixedLenFeature
        all_features = {
            "input_ids":    FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":   FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":  FixedLenFeature([max_seq_length], tf.int64),
            "d_input_ids":  FixedLenFeature([max_def_length], tf.int64),
            "d_input_mask": FixedLenFeature([max_def_length], tf.int64),
            "d_segment_ids": FixedLenFeature([max_def_length], tf.int64),
            "d_location_ids": FixedLenFeature([max_loc_length], tf.int64),
            "next_sentence_labels": FixedLenFeature([1], tf.int64),
            "masked_lm_positions": FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids": FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights": FixedLenFeature([max_predictions_per_seq], tf.float32),
            "lookup_idx": FixedLenFeature([1], tf.int64),
            "selected_word": FixedLenFeature([max_word_length], tf.int64),
            "ab_mapping": FixedLenFeature([def_per_batch], tf.int64),
            "ab_mapping_mask": FixedLenFeature([inner_batch_size * def_per_batch], tf.int64),
        }

        active_feature = ["input_ids", "input_mask", "segment_ids",
                          "d_input_ids", "d_input_mask", "d_location_ids",
                          "ab_mapping"]

        if flags.fixed_mask:
            active_feature.append("masked_lm_positions")
            active_feature.append("masked_lm_ids")
            active_feature.append("masked_lm_weights")
        if flags.train_op == "lookup":
            active_feature.append("masked_lm_positions")
            active_feature.append("masked_lm_ids")
            active_feature.append("lookup_idx")
        elif flags.train_op == "entry_prediction":
            active_feature.append("masked_lm_positions")
            active_feature.append("masked_lm_ids")
            active_feature.append("lookup_idx")

        if flags.use_d_segment_ids:
            active_feature.append("d_segment_ids")

        if flags.use_ab_mapping_mask:
            active_feature.append("ab_mapping_mask")

        if max_word_length > 0:
            active_feature.append("selected_word")

        selected_features = {k:all_features[k] for k in active_feature}

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        return tlm.training.input_fn_common.format_dataset(selected_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn
