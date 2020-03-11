from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import gather_index2d
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.dictionary.sense_selecting_dictionary_reader import get_batch_and_seq_length
from tlm.model.bert_with_label import BertModelWithLabelInner
from tlm.model.lm_objective import get_masked_lm_output, get_next_sentence_output
from tlm.model.masking import remove_special_mask, scatter_with_batch
from tlm.training.lm_model_fn import get_dummy_next_sentence_labels, align_checkpoint_for_lm, metric_fn_lm
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments
from trainer.tf_train_module_v2 import OomReportingHook



# input_id : [batch_size, max_sequence]
def one_by_one_masking(input_ids, input_masks, mask_token, n_trial):
    batch_size, seq_length = get_batch_and_seq_length(input_ids, 2)
    loc_dummy = tf.cast(tf.range(0, seq_length), tf.float32)
    loc_dummy = tf.tile(tf.expand_dims(loc_dummy, 0), [batch_size, 1])
    loc_dummy = remove_special_mask(input_ids, input_masks, loc_dummy)
    indices = tf.argsort(
        loc_dummy,
        axis=-1,
        direction='ASCENDING',
        stable=False,
        name=None
    )
    # [25, batch, 20]
    n_input_ids = tf.tile(input_ids, [n_trial, 1])
    lm_locations = tf.reshape(indices[:, :n_trial], [-1, 1])
    masked_lm_positions = lm_locations # [ batch*n_trial, max_predictions)
    masked_lm_ids = gather_index2d(n_input_ids, masked_lm_positions)
    masked_lm_weights = tf.ones_like(masked_lm_positions, dtype=tf.float32)
    masked_input_ids = scatter_with_batch(n_input_ids, masked_lm_positions, mask_token)
    return masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights


def model_fn_generative_predictor(model_config, train_config,
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
        next_sentence_labels = get_dummy_next_sentence_labels(input_ids)
        batch_size, seq_length = get_batch_and_seq_length(input_ids, 2)
        n_trial = seq_length - 20

        masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
            = one_by_one_masking(input_ids, input_mask, MASK_ID, n_trial)
        num_classes = train_config.num_classes
        n_repeat = num_classes * n_trial

        # [ num_classes * n_trial * batch_size, seq_length]
        repeat_masked_input_ids = tf.tile(masked_input_ids, [num_classes, 1])
        repeat_input_mask = tf.tile(input_mask, [n_repeat, 1])
        repeat_segment_ids = tf.tile(segment_ids, [n_repeat, 1])
        masked_lm_positions = tf.tile(masked_lm_positions, [num_classes, 1])
        masked_lm_ids = tf.tile(masked_lm_ids, [num_classes, 1])
        masked_lm_weights = tf.tile(masked_lm_weights, [num_classes, 1])
        next_sentence_labels = tf.tile(next_sentence_labels, [n_repeat, 1])

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        virtual_labels_ids = tf.tile(tf.expand_dims(tf.range(num_classes), 0), [1, batch_size * n_trial])
        virtual_labels_ids = tf.reshape(virtual_labels_ids, [-1, 1])

        print("repeat_masked_input_ids", repeat_masked_input_ids.shape)
        print("repeat_input_mask", repeat_input_mask.shape)
        print("virtual_labels_ids", virtual_labels_ids.shape)
        model = BertModelWithLabelInner(
            config=model_config,
            is_training=is_training,
            input_ids=repeat_masked_input_ids,
            input_mask=repeat_input_mask,
            token_type_ids=repeat_segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            label_ids=virtual_labels_ids,
        )
        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output_fn(
                 model_config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
                 model_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss

        # loss = -log(prob)
        # TODO compare log prob of each label

        per_case_loss = tf.reshape(masked_lm_example_loss, [num_classes, -1, batch_size])
        per_label_loss = tf.reduce_sum(per_case_loss, axis=1)
        bias = tf.zeros([3, 1])
        per_label_score = tf.transpose(-per_label_loss + bias , [1, 0])

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
                    "input_ids": input_ids,
                    "logits": per_label_score
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)


        return output_spec

    return model_fn
