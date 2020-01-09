
import tensorflow as tf

from data_generator.special_tokens import MASK_ID
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.masking import random_masking
from tlm.tlm.model_fn_try_all_loss import get_init_fn
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments


def model_fn_preserved_dim(bert_config, train_config):
    """Returns `model_fn` closure for TPUEstimator."""
    logging = tf_logging
    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        next_sentence_labels = features["next_sentence_labels"]

        seed = 0
        threshold = 1e-2
        logging.info("Doing All Masking")
        masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
            = random_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID, seed)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        prefix1 = "MaybeBERT"
        prefix2 = "MaybeNLI"

        with tf.compat.v1.variable_scope(prefix1):
            model = BertModel(
                    config=bert_config,
                    is_training=is_training,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=segment_ids,
                    use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )
            (masked_lm_loss,
             masked_lm_example_loss1, masked_lm_log_probs2) = get_masked_lm_output(
                     bert_config, model.get_sequence_output(), model.get_embedding_table(),
                     masked_lm_positions, masked_lm_ids, masked_lm_weights)
            all_layers1 = model.get_all_encoder_layers()

        with tf.compat.v1.variable_scope(prefix2):
            model = BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )
            all_layers2 = model.get_all_encoder_layers()

        preserved_infos = []
        for a_layer, b_layer in zip(all_layers1, all_layers2):
            layer_diff = a_layer - b_layer
            is_preserved = tf.less(tf.abs(layer_diff), threshold)
            preserved_infos.append(is_preserved)

        t = tf.cast(preserved_infos[1], dtype=tf.int32) #[batch_size, seq_len, dims]
        layer_1_count = tf.reduce_sum(t, axis=2)

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names, init_fn = get_init_fn(train_config,
                                                          tvars,
                                                          train_config.init_checkpoint,
                                                          prefix1,
                                                          train_config.second_init_checkpoint,
                                                          prefix2)
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)

        log_var_assignments(tvars, initialized_variable_names)

        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                "input_ids":input_ids,
                "layer_count":layer_1_count
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=None,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn