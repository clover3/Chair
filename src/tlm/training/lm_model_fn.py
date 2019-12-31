import tensorflow as tf

from data_generator.special_tokens import MASK_ID, PAD_ID
from models.transformer import bert_common_v2 as bert_common
from models.transformer import hyperparams
from models.transformer import optimization_v2 as optimization
from tf_util.tf_logging import tf_logging
from tlm.model.lm_objective import get_masked_lm_output, get_next_sentence_output
from tlm.model.masking import random_masking, biased_masking
from tlm.model.nli_ex_v2 import transformer_nli
from tlm.training.assignment_map import get_bert_assignment_map, get_cls_assignment, get_tlm_assignment_map_v2, \
    assignment_map_v2_to_v2, get_assignment_map_remap_from_v2
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments


def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, next_sentence_example_loss,
              next_sentence_log_probs, next_sentence_labels):
    """Computes the loss and accuracy of the model."""
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(
        input=masked_lm_log_probs, axis=-1, output_type=tf.int32)
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)
    masked_lm_mean_loss = tf.compat.v1.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights)

    next_sentence_log_probs = tf.reshape(
        next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
    next_sentence_predictions = tf.argmax(
        input=next_sentence_log_probs, axis=-1, output_type=tf.int32)
    next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
    next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
        labels=next_sentence_labels, predictions=next_sentence_predictions)
    next_sentence_mean_loss = tf.compat.v1.metrics.mean(
        values=next_sentence_example_loss)

    return {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
        "next_sentence_accuracy": next_sentence_accuracy,
        "next_sentence_loss": next_sentence_mean_loss,
    }


def metric_fn_lm(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights):
    """Computes the loss and accuracy of the model."""
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(
        input=masked_lm_log_probs, axis=-1, output_type=tf.int32)
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)
    masked_lm_mean_loss = tf.compat.v1.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights)

    return {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
    }


def align_checkpoint_for_lm(tvars,
                            checkpoint_type,
                            init_checkpoint,
                            second_init_checkpoint=None,
                            use_multiple_checkpoint=False):
    tf_logging.debug("align_checkpoint_for_lm")
    initialized_variable_names2 = {}
    if init_checkpoint:
        if not use_multiple_checkpoint:
            if checkpoint_type == "":
                assignment_fn = get_bert_assignment_map
            elif checkpoint_type == "v2":
                assignment_fn = assignment_map_v2_to_v2
            else:
                raise Exception("Undefined checkpoint exists")

            assignment_map, initialized_variable_names = assignment_fn(tvars, init_checkpoint)

            def init_fn():
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        else:
            if checkpoint_type == "nli_and_bert":
                assignment_map, initialized_variable_names = get_bert_assignment_map(tvars,
                                                                                     init_checkpoint)
                assignment_map2, initialized_variable_names2 = get_cls_assignment(tvars,
                                                                                  second_init_checkpoint)
            else:
                raise Exception("Undefined checkpoint exists")

            def init_fn():
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

                tf.compat.v1.train.init_from_checkpoint(second_init_checkpoint, assignment_map2)

    else:
        initialized_variable_names = {}
        def init_fn():
            pass
    return initialized_variable_names, initialized_variable_names2, init_fn


def model_fn_lm(bert_config, train_config, model_class):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        if "next_sentence_labels" in features:
            next_sentence_labels = features["next_sentence_labels"]
        else:
            next_sentence_labels = get_dummy_next_sentence_labels(input_ids)

        if mode == tf.estimator.ModeKeys.PREDICT:
                tf.random.set_seed(0)
                seed = 0
                print("Seed as zero")
        else:
                seed = None

        if not train_config.fixed_mask:
            tf_logging.info("Doing dynamic masking (random)")
            masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
                = random_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID, seed)
        else:
            tf_logging.info("Using masking from input")
            masked_input_ids = input_ids
            masked_lm_positions = features["masked_lm_positions"]
            masked_lm_ids = features["masked_lm_ids"]
            masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = model_class(
                config=bert_config,
                is_training=is_training,
                input_ids=masked_input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
                 bert_config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
                 bert_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss

        tvars = tf.compat.v1.trainable_variables()

        use_multiple_checkpoint = train_config.checkpoint_type == "nli_and_bert"
        initialized_variable_names, initialized_variable_names2, init_fn\
            = align_checkpoint_for_lm(tvars,
                                      train_config.checkpoint_type,
                                      train_config.init_checkpoint,
                                      train_config.second_init_checkpoint,
                                      use_multiple_checkpoint)

        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)

        log_var_assignments(tvars, initialized_variable_names, initialized_variable_names2)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer_from_config(total_loss, train_config)
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn_lm, [
                    masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights,
            ])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "input_ids":input_ids,
                    "masked_input_ids":masked_input_ids,
                    "masked_lm_ids":masked_lm_ids,
                    "masked_lm_example_loss":masked_lm_example_loss,
                    "masked_lm_positions":masked_lm_positions,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)


        return output_spec

    return model_fn


def get_nli_ex_model(input_ids, input_mask, segment_ids):
  method = 5
  hp = hyperparams.HPBert()
  voca_size = 30522

  model = transformer_nli(hp, input_ids, input_mask, segment_ids, voca_size, method, False)
  output = model.conf_logits

  return output


def get_nli_ex_model_segmented(input_ids, input_mask, segment_ids):
    method = 5
    hp = hyperparams.HPBert()
    voca_size = 30522
    sequence_shape = bert_common.get_shape_list2(input_ids)
    batch_size = sequence_shape[0]

    step = 200
    pad_len = 200 - 1 - (512 - (step * 2 - 1))
    def spread(t):
        cls_token = t[:,:1]
        pad = tf.ones([batch_size, pad_len], tf.dtypes.int32) * PAD_ID
        a = t[:, :step]
        b = tf.concat([cls_token, t[:,step:step*2-1]], axis=1)
        c = tf.concat([cls_token, t[:,step*2-1:], pad], axis=1)
        return tf.concat([a,b,c], axis=0)

    def collect(t):
        a = t[:batch_size]
        b = t[batch_size:batch_size *2, 1:]
        c = t[batch_size*2:, 1:-pad_len]
        return tf.concat([a,b,c], axis=1)

    model = transformer_nli(hp, spread(input_ids), spread(input_mask), spread(segment_ids), voca_size, method, False)
    output = model.conf_logits
    output = collect(output)
    return output


def get_dummy_next_sentence_labels(input_ids):
    sequence_shape = bert_common.get_shape_list2(input_ids)
    batch_size = sequence_shape[0]
    next_sentence_labels = tf.zeros([batch_size, 1], tf.int64)
    return next_sentence_labels


def model_fn_target_masking(bert_config, train_config, target_model_config, model_class, priority_model):
    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        if "next_sentence_labels" in features:
            next_sentence_labels = features["next_sentence_labels"]
        else:
            next_sentence_labels = get_dummy_next_sentence_labels(input_ids)
        tlm_prefix = "target_task"

        with tf.compat.v1.variable_scope(tlm_prefix):
            priority_score = tf.stop_gradient(priority_model(features))

        priority_score = priority_score * target_model_config.amp
        masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights\
            = biased_masking(input_ids,
                             input_mask,
                             priority_score,
                             target_model_config.alpha,
                             train_config.max_predictions_per_seq,
                             MASK_ID)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = model_class(
                config=bert_config,
                is_training=is_training,
                input_ids=masked_input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
                 bert_config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
                 bert_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss + next_sentence_loss

        all_vars = tf.compat.v1.all_variables()

        tf_logging.info("We assume priority model is from v2")

        if train_config.checkpoint_type == "v2":
            assignment_map, initialized_variable_names = assignment_map_v2_to_v2(all_vars, train_config.init_checkpoint)
            assignment_map2, initialized_variable_names2 = get_assignment_map_remap_from_v2(all_vars, tlm_prefix,
                                                                                            train_config.second_init_checkpoint)
        else:
            assignment_map, assignment_map2, initialized_variable_names \
                                            = get_tlm_assignment_map_v2(all_vars,
                                              tlm_prefix,
                                              train_config.init_checkpoint,
                                              train_config.second_init_checkpoint)
            initialized_variable_names2 = None

        def init_fn():
            if train_config.init_checkpoint:
                tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
            if train_config.second_init_checkpoint:
                tf.compat.v1.train.init_from_checkpoint(train_config.second_init_checkpoint, assignment_map2)

        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)

        tvars = [v for v in all_vars if not v.name.startswith(tlm_prefix)]
        log_var_assignments(tvars, initialized_variable_names, initialized_variable_names2)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer_from_config(total_loss, train_config, tvars)
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
            predictions = {
                    "input_ids": input_ids,
                    "masked_input_ids": masked_input_ids,
                    "priority_score": priority_score,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn



