import copy

from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import get_shape_list
from models.transformer.optimization_v2 import create_simple_optimizer
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2
from tlm.training.classification_model_fn import get_init_fn
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments


def shift(v):
    return tf.math.floormod(v+2, 3)


def classification_metric_fn(pred, label, is_real_example):
    """Computes the loss and accuracy of the model."""
    label = tf.reshape(label, [-1])
    accuracy = tf.compat.v1.metrics.accuracy(
      labels=label, predictions=pred, weights=is_real_example)

    precision = tf.compat.v1.metrics.precision(
      labels=label, predictions=pred, weights=is_real_example)

    recall = tf.compat.v1.metrics.recall(
        labels=label, predictions=pred, weights=is_real_example)

    return {
      "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def model_fn_pairwise_ranking(model_config, train_config, model_class, special_flags=[], override_prediction_fn=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    tf_logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    q_input_ids_1 = features["q_input_ids_1"]
    q_input_mask_1 = features["q_input_mask_1"]
    d_input_ids_1 = features["d_input_ids_1"]
    d_input_mask_1 = features["d_input_mask_1"]

    q_input_ids_2 = features["q_input_ids_2"]
    q_input_mask_2 = features["q_input_mask_2"]
    d_input_ids_2 = features["d_input_ids_2"]
    d_input_mask_2 = features["d_input_mask_2"]
    
    q_input_ids = tf.stack([q_input_ids_1, q_input_ids_2], axis=0)
    q_input_mask = tf.stack([q_input_mask_1, q_input_mask_2], axis=0)
    q_segment_ids = tf.zeros_like(q_input_ids, tf.int32)

    d_input_ids = tf.stack([d_input_ids_1, d_input_ids_2], axis=0)
    d_input_mask = tf.stack([d_input_mask_1, d_input_mask_2], axis=0)
    d_segment_ids = tf.zeros_like(d_input_ids, tf.int32)

    label_ids = features["label_ids"]
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    with tf.compat.v1.variable_scope("query"):
        model_q = model_class(
            config=model_config,
            is_training=is_training,
            input_ids=q_input_ids,
            input_mask=q_input_mask,
            token_type_ids=q_segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )

    with tf.compat.v1.variable_scope("document"):
        model_d = model_class(
            config=model_config,
            is_training=is_training,
            input_ids=d_input_ids,
            input_mask=d_input_mask,
            token_type_ids=d_segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )
    pooled_q = model_q.get_pooled_output()
    pooled_d = model_d.get_pooled_output()

    logits = tf.matmul(pooled_q, pooled_d, transpose_b=True)
    y = tf.cast(label_ids, tf.float32) * 2 - 1
    losses = tf.maximum(1.0 - logits * y, 0)
    loss = tf.reduce_mean(losses)

    pred = tf.cast(logits > 0, tf.int32)

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
            train_op = optimization.create_optimizer_from_config(loss, train_config, tvars)
        output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics = (classification_metric_fn, [
            pred, label_ids, is_real_example
        ])
        output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
    else:
        predictions = {
                "q_input_ids": q_input_ids,
                "d_input_ids": d_input_ids,
                "score": logits
        }

        useful_inputs = ["data_id", "input_ids2", "data_ids"]
        for input_name in useful_inputs:
            if input_name in features:
                predictions[input_name] = features[input_name]
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                scaffold_fn=scaffold_fn)

    return output_spec
  return model_fn


def model_fn_pointwise_ranking(model_config, train_config, model_class, special_flags=[], override_prediction_fn=None):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        q_input_ids = features["q_input_ids"]
        q_input_mask = features["q_input_mask"]
        d_input_ids = features["d_input_ids"]
        d_input_mask = features["d_input_mask"]

        input_shape = get_shape_list(q_input_ids, expected_rank=2)
        batch_size = input_shape[0]

        doc_length = model_config.max_doc_length
        num_docs = model_config.num_docs

        d_input_ids_unpacked = tf.reshape(d_input_ids, [-1, num_docs, doc_length])
        d_input_mask_unpacked = tf.reshape(d_input_mask, [-1, num_docs, doc_length])

        d_input_ids_flat = tf.reshape(d_input_ids_unpacked, [-1, doc_length])
        d_input_mask_flat = tf.reshape(d_input_mask_unpacked, [-1, doc_length])

        q_segment_ids = tf.zeros_like(q_input_ids, tf.int32)
        d_segment_ids = tf.zeros_like(d_input_ids_flat, tf.int32)

        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            q_model_config = copy.deepcopy(model_config)
            q_model_config.max_seq_length = model_config.max_sent_length
            model_q = model_class(
                config=model_config,
                is_training=is_training,
                input_ids=q_input_ids,
                input_mask=q_input_mask,
                token_type_ids=q_segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            d_model_config = copy.deepcopy(model_config)
            d_model_config.max_seq_length = model_config.max_doc_length
            model_d = model_class(
                config=model_config,
                is_training=is_training,
                input_ids=d_input_ids_flat,
                input_mask=d_input_mask_flat,
                token_type_ids=d_segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            )
        pooled_q = model_q.get_pooled_output() # [batch, vector_size]
        pooled_d_flat = model_d.get_pooled_output() # [batch, num_window, vector_size]

        pooled_d = tf.reshape(pooled_d_flat, [batch_size, num_docs, -1])
        pooled_q_t = tf.expand_dims(pooled_q, 1)
        pooled_d_t = tf.transpose(pooled_d, [0, 2, 1])
        all_logits = tf.matmul(pooled_q_t, pooled_d_t) # [batch, 1, num_window]
        if "hinge_all" in special_flags:
            apply_loss_modeing = hinge_all
        elif "sigmoid_all" in special_flags:
            apply_loss_modeing = sigmoid_all
        else:
            apply_loss_modeing = hinge_max
        logits, loss = apply_loss_modeing(all_logits, label_ids)
        pred = tf.cast(logits > 0, tf.int32)

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
                train_op = optimization.create_optimizer_from_config(loss, train_config, tvars)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (classification_metric_fn, [
                pred, label_ids, is_real_example
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
        else:
            predictions = {
                "q_input_ids": q_input_ids,
                "d_input_ids": d_input_ids,
                "logits": logits
            }

            useful_inputs = ["data_id", "input_ids2", "data_ids"]
            for input_name in useful_inputs:
                if input_name in features:
                    predictions[input_name] = features[input_name]
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def hinge_all(all_logits, label_ids):
    print('all_logits', all_logits)
    # logits = tf.reduce_max(all_logits, axis=2)
    print('logits', all_logits)
    y = tf.cast(label_ids, tf.float32) * 2 - 1
    print('label_ids', label_ids)
    print('y', y)
    y_expand = tf.expand_dims(y, 2)
    print('y_expand')
    t = all_logits * y_expand
    losses = tf.maximum(1.0 - t, 0)
    loss = tf.reduce_mean(losses)
    logits = tf.reduce_mean(all_logits, axis=2)
    return logits, loss


def hinge_max(all_logits, label_ids):
    print('all_logits', all_logits)
    logits = tf.reduce_max(all_logits, axis=2)
    print('logits', all_logits)
    y = tf.cast(label_ids, tf.float32) * 2 - 1
    print('label_ids', label_ids)
    print('y', y)
    t = logits * y
    losses = tf.maximum(1.0 - t, 0)
    loss = tf.reduce_mean(losses)
    logits = tf.reduce_mean(all_logits, axis=2)
    return logits, loss


def sigmoid_all(all_logits, label_ids):
    print('all_logits', all_logits)
    print('logits', all_logits)
    batch_size, _, num_seg = get_shape_list(all_logits)
    lable_ids_tile = tf.cast(tf.tile(tf.expand_dims(label_ids, 2), [1, 1, num_seg]), tf.float32)
    print('label_ids', label_ids)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=all_logits, labels=lable_ids_tile)
    loss = tf.reduce_mean(losses)

    probs = tf.nn.sigmoid(all_logits)
    logits = tf.reduce_mean(probs, axis=2)
    return logits, loss






