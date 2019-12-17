
import tensorflow as tf

from data_generator.special_tokens import MASK_ID
from tlm.model.base import BertModel, gather_index2d
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.masking import remove_special_mask, scatter_with_batch
from tlm.training.assignment_map import get_assignment_map_remap_from_v1, \
    get_assignment_map_remap_from_v2
from trainer.get_param_num import get_param_num


# input_id : [batch_size, max_sequence]
def planned_masking(input_ids, input_masks, max_predictions_per_seq, mask_token, n_trial):
    rand = tf.random.uniform(
        input_ids.shape,
        minval=0,
        maxval=1,
        dtype=tf.dtypes.float32,
        seed=0,
        name=None
    )
    rand = remove_special_mask(input_ids, input_masks, rand)
    random_seq = tf.argsort(
        rand,
        axis=-1,
        direction='DESCENDING',
        stable=False,
        name=None
    )

    lm_locations_list = []
    for i in range(n_trial):
        st = i * max_predictions_per_seq
        ed = (i+1) * max_predictions_per_seq
        lm_locations = random_seq[:, st:ed]
        lm_locations_list.append(lm_locations)
    # [25, batch, 20]
    n_input_ids = tf.tile(input_ids, [n_trial, 1])
    masked_lm_positions = tf.concat(lm_locations_list, axis=0) # [ batch*n_trial, max_predictions)
    masked_lm_ids = gather_index2d(n_input_ids , masked_lm_positions)
    masked_lm_weights = tf.ones_like(masked_lm_positions, dtype=tf.float32)
    masked_input_ids = scatter_with_batch(n_input_ids, masked_lm_positions, mask_token)
    return masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights


def model_fn_try_all_loss(bert_config, train_config, logging):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        next_sentence_labels = features["next_sentence_labels"]

        n_trial = 25

        logging.info("Doing All Masking")
        masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
            = planned_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID, n_trial)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        repeat_input_mask = tf.tile(input_mask, [n_trial, 1])
        repeat_segment_ids = tf.tile(segment_ids, [n_trial, 1])
        prefix1 = "MaybeBERT"
        prefix2 = "MaybeBFN"

        with tf.compat.v1.variable_scope(prefix1):
            model = BertModel(
                    config=bert_config,
                    is_training=is_training,
                    input_ids=masked_input_ids,
                    input_mask=repeat_input_mask,
                    token_type_ids=repeat_segment_ids,
                    use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )
            (masked_lm_loss,
             masked_lm_example_loss1, masked_lm_log_probs2) = get_masked_lm_output(
                     bert_config, model.get_sequence_output(), model.get_embedding_table(),
                     masked_lm_positions, masked_lm_ids, masked_lm_weights)

        with tf.compat.v1.variable_scope(prefix2):
            model = BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=masked_input_ids,
                input_mask=repeat_input_mask,
                token_type_ids=repeat_segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )

            (masked_lm_loss,
             masked_lm_example_loss2, masked_lm_log_probs2) = get_masked_lm_output(
                bert_config, model.get_sequence_output(), model.get_embedding_table(),
                masked_lm_positions, masked_lm_ids, masked_lm_weights)

        n_mask = train_config.max_predictions_per_seq

        def reform(t):
            t = tf.reshape(t, [n_trial, -1, n_mask])
            t = tf.transpose(t, [1,0,2])
            return t
        grouped_positions = reform(masked_lm_positions)
        grouped_loss1 = reform(masked_lm_example_loss1)
        grouped_loss2 = reform(masked_lm_example_loss2)
        tvars = tf.compat.v1.trainable_variables()

        scaffold_fn = None
        initialized_variable_names, init_fn = get_init_fn(tvars,
                                                          train_config.init_checkpoint,
                                                          prefix1,
                                                          train_config.second_init_checkpoint,
                                                          prefix2)
        if train_config.use_tpu:
            def tpu_scaffold():
                init_fn()
                return tf.compat.v1.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            init_fn()

        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("    name = %s, shape = %s%s", var.name, var.shape,
                                            init_string)
        logging.info("Total parameters : %d" % get_param_num())

        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                    "input_ids": input_ids,
                    "input_mask": input_mask,
                    "segment_ids": segment_ids,
                    "grouped_positions": grouped_positions,
                    "grouped_loss1": grouped_loss1,
                    "grouped_loss2": grouped_loss2,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=None,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


# init_checkpoint : BERT (v1)
# second_init_checkpoint : v2
def get_init_fn(tvars, init_checkpoint, remap_prefix, second_init_checkpoint, remap_prefix2):
    assignment_map, initialized_variable_names \
        = get_assignment_map_remap_from_v1(tvars, remap_prefix, init_checkpoint)
    assignment_map2, initialized_variable_names2 \
        = get_assignment_map_remap_from_v2(tvars, remap_prefix2, second_init_checkpoint)
    for k, v in initialized_variable_names2.items():
        initialized_variable_names[k] = v

    def init_fn():
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.compat.v1.train.init_from_checkpoint(second_init_checkpoint, assignment_map2)

    return initialized_variable_names, init_fn

