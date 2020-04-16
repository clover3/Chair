from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.lm_objective import get_masked_lm_output, get_next_sentence_output
from tlm.model.masking import random_masking
from tlm.training.lm_model_fn import get_dummy_next_sentence_labels, align_checkpoint_for_lm, metric_fn_lm
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments
from trainer.tf_train_module_v2 import OomReportingHook





def model_fn_lm(model_config, train_config, model_class,
                get_masked_lm_output_fn=get_masked_lm_output, feed_feature=False):
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

        if not feed_feature:
            model = model_class(
                    config=model_config,
                    is_training=is_training,
                    input_ids=masked_input_ids,
                    input_mask=input_mask,
                    token_type_ids=segment_ids,
                    use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )
        else:
            model = model_class(
                config=model_config,
                is_training=is_training,
                input_ids=masked_input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
                features=features,
            )

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output_fn(
                 model_config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
                 model_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss

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
            train_op = optimization.create_optimizer_from_config(total_loss, train_config, model.get_trainable_vars())
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
            grad = tf.gradients(masked_lm_example_loss, model.embedding_table_2)[0]
            predictions = {
                    "input_ids":input_ids,
                    "masked_input_ids":masked_input_ids,
                    "masked_lm_ids":masked_lm_ids,
                    "masked_lm_example_loss":masked_lm_example_loss,
                    "masked_lm_positions":masked_lm_positions,
                    "grad": grad,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)


        return output_spec

    return model_fn
