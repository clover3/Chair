import tensorflow as tf

import tlm.training.dict_model_fn as dict_model_fn
import tlm.training.input_fn
import tlm.training.input_fn_common
from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.masking import random_masking
from tlm.training.grad_accumulation import get_accumulated_optimizer_from_config
from tlm.training.lm_model_fn import metric_fn, get_assignment_map_as_is
from trainer.get_param_num import get_param_num


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
            eval_metrics = (metric_fn, [
                    masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels
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
                        #"input_ids": input_ids,
                        "d_input_ids": d_input_ids,
                        "masked_lm_positions": masked_lm_positions,
                        "gradients": gradient,
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
