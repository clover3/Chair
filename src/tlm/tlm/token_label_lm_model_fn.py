from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import gather_index2d, get_shape_list2
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.lm_objective import get_masked_lm_output, get_next_sentence_output
from tlm.model.masking import random_masking
from tlm.training.lm_model_fn import get_dummy_next_sentence_labels, align_checkpoint_for_lm
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments
from trainer.tf_train_module_v2 import OomReportingHook

LABEL_UNK = 10
LABEL_0 = 11
LABEL_1 = 12
LABEL_2 = 13

def scatter_multiple(input_ids, indice, update_vals):
    batch_size = get_shape_list2(input_ids)[0]
    seq_length = get_shape_list2(input_ids)[1]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    indices = tf.reshape(indice + flat_offsets, [-1, 1])
    tensor = tf.reshape(input_ids, [batch_size*seq_length])

    updates = tf.reshape(update_vals, [-1])
    flat_output = tf.tensor_scatter_nd_update(tensor, indices, updates)
    return tf.reshape(flat_output, [batch_size, seq_length])


def get_label_indices(input_ids):
    test_label = [LABEL_0, LABEL_1, LABEL_2]
    test_label_mask = tf.cast(tf.zeros_like(input_ids), tf.bool)
    for token in test_label:
        test_label_mask = tf.logical_or(tf.equal(input_ids, token), test_label_mask)

    _, masked_lm_positions = tf.math.top_k(
        tf.cast(test_label_mask, tf.float32),
        k=1,
        sorted=False,
        name="masking_top_k"
    )
    is_test_inst_bool = tf.reduce_any(test_label_mask, axis=1)
    is_test_inst = tf.cast(tf.reduce_any(test_label_mask, axis=1), tf.float32)

    masked_label_ids = gather_index2d(input_ids, masked_lm_positions)

    is_test_inst_int = tf.cast(is_test_inst, tf.int32)
    not_is_test_inst_int = tf.cast(tf.logical_not(is_test_inst_bool), tf.int32)
    scatter_vals = LABEL_UNK * is_test_inst_int\
                   + tf.reshape(masked_label_ids, [-1]) * not_is_test_inst_int
    masked_input_ids = scatter_multiple(input_ids, masked_lm_positions, scatter_vals)
    return masked_input_ids, masked_lm_positions, masked_label_ids, is_test_inst


def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights,
                masked_lm_example_loss_label, masked_lm_log_probs_label,
              masked_lm_ids_label, masked_lm_weights_label,
              ):
    info1 = sub_metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights)

    info2 = sub_metric_fn(masked_lm_example_loss_label, masked_lm_log_probs_label, masked_lm_ids_label,
                  masked_lm_weights_label)

    for key, value in info2.items():
        info1[key+"_label"] = value
    return info1


def sub_metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights,
              ):
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


def model_fn_lm(model_config, train_config, model_class,
                get_masked_lm_output_fn=get_masked_lm_output):
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

        tf_logging.info("Doing dynamic masking (random)")
        special_tokens = [LABEL_UNK, LABEL_0, LABEL_1, LABEL_2]
        masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
            = random_masking(input_ids,
                             input_mask,
                             train_config.max_predictions_per_seq,
                             MASK_ID,
                             seed,
                             special_tokens)

        masked_input_ids, masked_lm_positions_label, masked_label_ids_label, is_test_inst \
            = get_label_indices(masked_input_ids)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = model_class(
                config=model_config,
                is_training=is_training,
                input_ids=masked_input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output_fn(
                 model_config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)

        with tf.compat.v1.variable_scope("label_token"):
            (masked_lm_loss_label,
             masked_lm_example_loss_label, masked_lm_log_probs_label) = get_masked_lm_output_fn(
                     model_config, model.get_sequence_output(), model.get_embedding_table(),
                     masked_lm_positions_label, masked_label_ids_label, is_test_inst)
        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
                 model_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss + masked_lm_loss_label * model_config.ratio

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names, initialized_variable_names2, init_fn\
            = align_checkpoint_for_lm(tvars,
                                      train_config.checkpoint_type,
                                      train_config.init_checkpoint,
                                      train_config.second_init_checkpoint,
                                      )

        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)

        log_var_assignments(tvars, initialized_variable_names, initialized_variable_names2)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer_from_config(total_loss, train_config)
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    training_hooks=[OomReportingHook()],

                scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn, [
                    masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights,
                    masked_lm_example_loss_label, masked_lm_log_probs_label, masked_label_ids_label, is_test_inst
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
                    "masked_lm_example_loss_label":masked_lm_example_loss_label,
                    "masked_lm_log_probs_label":masked_lm_log_probs_label,
                    "masked_label_ids_label":masked_label_ids_label,
                    "is_test_inst":is_test_inst,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)


        return output_spec

    return model_fn
