from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import dropout, get_shape_list2
from models.transformer.optimization_v2 import create_simple_optimizer
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.base import mimic_pooling
from tlm.model.masking import scatter_with_batch
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments, classification_metric_fn, \
    reweight_zero


def delete_tokens(input_ids, n_trial, shift):
    delete_location = []
    n_block_size = 1
    for i in range(n_trial):
        st = shift + i * n_block_size
        ed = shift + (i+1) * n_block_size
        row = []
        for j in range(st, ed):
            row.append(j)

        delete_location.append(row)
    print(delete_location)
    batch_size, _ = get_shape_list2(input_ids)

    # [n_trial, 1]
    delete_location = tf.constant(delete_location, tf.int32)
    # [1, n_trial, 1]
    delete_location = tf.expand_dims(delete_location, 0)
    # [batch_size, n_trial, 1]
    delete_location = tf.tile(delete_location, [batch_size, 1, 1])
    # [n_trial, batch, 1]
    delete_location = tf.transpose(delete_location, [1, 0, 2])
    # [n_trial * batch, 1]
    delete_location = tf.reshape(delete_location, [batch_size * n_trial, -1])
    n_input_ids = tf.tile(input_ids, [n_trial, 1])
    masked_input_ids = scatter_with_batch(n_input_ids, delete_location, MASK_ID)
    return masked_input_ids


def model_fn_sensitivity(bert_config, train_config, model_class, special_flags=[], override_prediction_fn=None):
    """Returns `model_fn` closure for TPUEstimator."""
    shift_str = special_flags[0]
    shift = int(shift_str)
    if len(special_flags) > 1:
        n_trial = int(special_flags[1])
    else:
        n_trial = 20
    print(f"Shift={shift} n_trial={n_trial}")
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
          tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        input_ids_masked = delete_tokens(input_ids, n_trial, shift)

        input_ids_all = tf.concat([input_ids, input_ids_masked], axis=0)
        input_mask_all = tf.tile(input_mask, [n_trial+1, 1])
        segment_ids_all = tf.tile(segment_ids, [n_trial + 1, 1])

        if mode == tf.estimator.ModeKeys.PREDICT:
            label_ids = tf.ones([input_ids.shape[0]], dtype=tf.int32)
        else:
            label_ids = features["label_ids"]
            label_ids = tf.reshape(label_ids, [-1])

        label_ids = tf.tile(label_ids, [n_trial+1])

        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)


        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if "feed_features" in special_flags:
            model = model_class(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids_all,
                input_mask=input_mask_all,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
                features=features,
            )
        else:
            model = model_class(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids_all,
                input_mask=input_mask_all,
                token_type_ids=segment_ids_all,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )
        if "new_pooling" in special_flags:
            pooled = mimic_pooling(model.get_sequence_output(), bert_config.hidden_size, bert_config.initializer_range)
        else:
            pooled = model.get_pooled_output()

        if train_config.checkpoint_type != "bert_nli" and train_config.use_old_logits:
            tf_logging.info("Use old version of logistic regression")
            if is_training:
                pooled = dropout(pooled, 0.1)
            logits = tf.keras.layers.Dense(train_config.num_classes, name="cls_dense")(pooled)
        else:
            tf_logging.info("Use fixed version of logistic regression")
            output_weights = tf.compat.v1.get_variable(
                "output_weights", [train_config.num_classes, bert_config.hidden_size],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
            )

            output_bias = tf.compat.v1.get_variable(
                "output_bias", [train_config.num_classes],
                initializer=tf.compat.v1.zeros_initializer()
            )

            if is_training:
                pooled = dropout(pooled, 0.1)

            logits = tf.matmul(pooled, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)

        if "bias_loss" in special_flags:
            tf_logging.info("Using special_flags : bias_loss")
            loss_arr = reweight_zero(label_ids, loss_arr)

        loss = tf.reduce_mean(input_tensor=loss_arr)
        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}

        scaffold_fn = None
        if train_config.init_checkpoint:
          initialized_variable_names, init_fn = get_init_fn(train_config, tvars)
          scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)

        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            if "simple_optimizer" in special_flags:
                tf_logging.info("using simple optimizer")
                train_op = create_simple_optimizer(loss, train_config.learning_rate, train_config.use_tpu)
            else:
                if "ask_tvar" in special_flags:
                    tvars = model.get_trainable_vars()
                else:
                    tvars = None
                train_op = optimization.create_optimizer_from_config(loss, train_config, tvars)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (classification_metric_fn, [
                logits, label_ids, is_real_example
            ])
            output_spec = TPUEstimatorSpec(mode=model, loss=loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "input_ids": input_ids,
                    "logits": logits
            }

            useful_inputs = ["data_id", "input_ids2", "data_ids"]
            for input_name in useful_inputs:
                if input_name in features:
                    predictions[input_name] = features[input_name]

            if override_prediction_fn is not None:
                predictions = override_prediction_fn(predictions, model)

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec
    return model_fn
