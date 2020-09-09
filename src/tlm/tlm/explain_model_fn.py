
import tensorflow as tf

from models.transformer import optimization_v2 as optimization
from tlm.explain_payload_gen import candidate_gen
from tlm.model.base import BertModel, get_shape_list2
from tlm.training.assignment_map import get_init_fn_for_two_checkpoints
from tlm.training.model_fn_common import log_var_assignments
from trainer.tf_module import correlation_coefficient_loss


def select_best(best_run, #[batch_size, num_class]
                indice, # [batch_size, n_trial]
                length_arr):

    best_run_ex = tf.expand_dims(best_run, 2)
    d = tf.zeros_like(best_run_ex, tf.int64)
    best_run_idx = tf.concat([d, best_run_ex], axis=2)

    def select(arr):
        arr_ex = tf.expand_dims(arr, 1)
        return tf.gather_nd(arr_ex, best_run_idx, batch_dims=1)

    good_deletion_idx = select(indice)
    good_deletion_length = select(length_arr)
    return good_deletion_idx, good_deletion_length



def model_fn_explain(bert_config, train_config, logging):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        batch_size, seq_len = get_shape_list2(input_ids)
        n_trial = 5

        logging.info("Doing All Masking")
        new_input_ids, new_segment_ids, new_input_mask, indice, length_arr = \
            candidate_gen(input_ids, input_mask, segment_ids, n_trial)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        prefix_cls = "classification"
        prefix_explain = "explain"
        all_input_ids = tf.concat([input_ids, new_input_ids], axis=0)
        all_segment_ids = tf.concat([segment_ids, new_segment_ids], axis=0)
        all_input_mask = tf.concat([input_mask, new_input_mask], axis=0)
        with tf.compat.v1.variable_scope(prefix_cls):
            model = BertModel(
                    config=bert_config,
                    is_training=is_training,
                    input_ids=all_input_ids,
                    input_mask=all_input_mask,
                    token_type_ids=all_segment_ids,
                    use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )
            output_weights = tf.compat.v1.get_variable(
                "output_weights", [train_config.num_classes, bert_config.hidden_size],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
            )

            output_bias = tf.compat.v1.get_variable(
                "output_bias", [train_config.num_classes],
                initializer=tf.compat.v1.zeros_initializer()
            )
            pooled = model.get_pooled_output()
            raw_logits = tf.matmul(pooled, output_weights, transpose_b=True)
            logits = tf.stop_gradient(raw_logits)
            cls_logits = tf.nn.bias_add(logits, output_bias)
            cls_probs = tf.nn.softmax(cls_logits)

            orig_probs = cls_probs[:batch_size]
            new_probs = tf.reshape(cls_probs[batch_size:], [batch_size, n_trial, -1])

            best_run, informative = get_informative(new_probs, orig_probs)
            # informative.shape= [batch_size, num_clases]
            best_del_idx, best_del_len = select_best(best_run, indice, length_arr)

            signal_label = get_mask(best_del_idx, best_del_len, seq_len)

        with tf.compat.v1.variable_scope(prefix_explain):
            model = BertModel(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )
            seq = model.get_sequence_output()

            output_weights = tf.compat.v1.get_variable(
                "output_weights", [train_config.num_classes, bert_config.hidden_size],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
            )

            output_bias = tf.compat.v1.get_variable(
                "output_bias", [train_config.num_classes],
                initializer=tf.compat.v1.zeros_initializer()
            )
            logits = tf.matmul(seq, output_weights, transpose_b=True)
            ex_logits = tf.nn.bias_add(logits, output_bias) # [batch, seq_len, num_class]

        ex_logits_flat = tf.reshape(tf.transpose(ex_logits, [0,2,1]), [-1, seq_len])
        signal_label_flat = tf.cast(tf.reshape(signal_label, [-1, seq_len]), tf.float32)
        losses_per_clas_flat = correlation_coefficient_loss(signal_label_flat, ex_logits_flat) # [batch_size, num_class]
        losses_per_clas = tf.reshape(losses_per_clas_flat, [batch_size, -1])
        losses_per_clas = losses_per_clas * tf.cast(informative, tf.float32)
        losses = tf.reduce_mean(losses_per_clas, axis=1)
        loss = tf.reduce_mean(losses)

        tvars = tf.compat.v1.trainable_variables()

        scaffold_fn = None
        initialized_variable_names, init_fn = get_init_fn_for_two_checkpoints(train_config,
                                                                              tvars,
                                                                              train_config.init_checkpoint,
                                                                              prefix_explain,
                                                                              train_config.second_init_checkpoint,
                                                                              prefix_cls)
        if train_config.use_tpu:
            def tpu_scaffold():
                init_fn()
                return tf.compat.v1.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            init_fn()

        log_var_assignments(tvars, initialized_variable_names)

        output_spec = None
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer_from_config(loss, train_config)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                    "input_ids": input_ids,
                    "ex_logits": ex_logits,
                    "logits": logits,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=None,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def get_informative(new_probs, orig_probs, threshold=0.5):
    orig_probs = tf.expand_dims(orig_probs, 1)
    diff_probs = orig_probs - new_probs
    diff_probs_t = tf.transpose(diff_probs, [0, 2, 1]) # [batch_size, num_classes, num_trials]
    best_run = tf.argmax(diff_probs_t, axis=2)  # [batch_size, num_classes]
    best_run_ = tf.expand_dims(best_run, 2)

    drops = tf.gather_nd(diff_probs_t, best_run_, batch_dims=2) # [batch_size, num_clases]
    informative = tf.less(threshold, drops)

    return best_run, informative


def get_mask(start, length, max_len):
    batch_size, _ = get_shape_list2(start)
    end = start + length
    a = tf.expand_dims(tf.expand_dims(tf.range(max_len), 0), 0)

    start_i = tf.expand_dims(start, 2)
    end_i = tf.expand_dims(end, 2)
    mask = tf.logical_and(tf.less_equal(start_i, a), tf.less(a, end_i))
    return mask
