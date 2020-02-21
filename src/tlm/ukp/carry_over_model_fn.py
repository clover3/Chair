import numpy
import tensorflow as tf

from data_generator.special_tokens import MASK_ID
from models.transformer.bert_common_v2 import gather_index2d, get_shape_list2
from tf_util.tf_logging import tf_logging
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.masking import remove_special_mask, scatter_with_batch
from tlm.training.lm_model_fn import get_dummy_next_sentence_labels, align_checkpoint_for_lm
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments


def random_masking(input_ids, input_masks, n_sample, mask_token, special_tokens=None):
    a_seg_len = 459

    part_cls = numpy.zeros([1])
    part_a_seg = numpy.random.random(a_seg_len)
    part_remain = numpy.zeros([512 - a_seg_len - 1])
    t = numpy.concatenate((part_cls, part_a_seg, part_remain))
    batch_size, _ = get_shape_list2(input_ids)
    base_random = tf.expand_dims(tf.constant(t, tf.float32), 0)
    rand = tf.tile(base_random, [batch_size, 1])
    print(rand.shape)

    if special_tokens is None:
        special_tokens = []
    rand = remove_special_mask(input_ids, input_masks, rand, special_tokens)
    _, indice = tf.math.top_k(
        rand,
        k=n_sample,
        sorted=False,
        name="masking_top_k"
    )
    masked_lm_positions = indice # [batch, n_samples]
    masked_lm_ids = gather_index2d(input_ids, masked_lm_positions)
    masked_lm_weights = tf.ones_like(masked_lm_positions, dtype=tf.float32)
    masked_input_ids = scatter_with_batch(input_ids, indice, mask_token)
    return masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights




def model_fn_lm(model_config, train_config, model_class):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        instance_id = features["instance_id"]
        next_sentence_labels = get_dummy_next_sentence_labels(input_ids)

        tf_logging.info("Doing dynamic masking (random)")
        masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
            = random_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID)

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
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(model_config,
                model.get_sequence_output(), model.get_embedding_table(),
                masked_lm_positions, masked_lm_ids, masked_lm_weights)


        total_loss = masked_lm_loss

        tvars = tf.compat.v1.trainable_variables()

        def is_multiple_checkpoint(checkpoint_type):
            return checkpoint_type in ["v2_and_bert" , "nli_and_bert"]
        use_multiple_checkpoint = is_multiple_checkpoint(train_config.checkpoint_type)
        initialized_variable_names, initialized_variable_names2, init_fn\
            = align_checkpoint_for_lm(tvars,
                                      train_config.checkpoint_type,
                                      train_config.init_checkpoint,
                                      train_config.second_init_checkpoint,
                                      use_multiple_checkpoint)

        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names, initialized_variable_names2)

        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                    "input_ids": input_ids,
                    "masked_lm_example_loss": masked_lm_example_loss,
                    "instance_id": instance_id
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)


        return output_spec

    return model_fn
