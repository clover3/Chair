from data_generator.special_tokens import CLS_ID, SEP_ID
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import dropout, get_shape_list2
from models.transformer.optimization_v2 import create_simple_optimizer
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.base import mimic_pooling
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments, classification_metric_fn, \
    reweight_zero


def iterate_over(query, doc, doc_mask, total_doc_len, segment_len, step_size):
    query_input_mask = tf.ones_like(query, tf.int32)
    query_segment_ids = tf.zeros_like(query, tf.int32)
    batch_size, _ = get_shape_list2(query)
    idx = 0
    input_ids_list = []
    input_masks_list = []
    input_segments_list = []
    n_segment = 0
    edge_shape = [batch_size, 1]
    cls_arr = tf.ones(edge_shape, tf.int32) * CLS_ID
    sep_arr = tf.ones(edge_shape, tf.int32) * SEP_ID
    edge_one = tf.ones(edge_shape, tf.int32)
    edge_zero = tf.zeros(edge_shape, tf.int32)

    while idx < total_doc_len:
        st = idx
        ed = idx + segment_len
        pad_len = ed - total_doc_len if ed > total_doc_len else 0
        padding = tf.zeros([batch_size, pad_len], tf.int32)
        doc_seg_input_ids = tf.concat([doc[:, st: ed], sep_arr, padding], axis=1)
        doc_seg_input_mask = tf.concat([doc_mask[:, st: ed], edge_one, padding], axis=1)
        doc_seg_segment_ids = tf.ones_like(doc_seg_input_ids, tf.int32) * doc_seg_input_mask

        input_ids = tf.concat([cls_arr, query, sep_arr, doc_seg_input_ids], axis=1)
        input_mask = tf.concat([edge_one, query_input_mask, edge_one, doc_seg_input_mask], axis=1)
        segment_ids = tf.concat([edge_zero, query_segment_ids, edge_zero, doc_seg_segment_ids], axis=1)

        input_ids_list.append(input_ids)
        input_masks_list.append(input_mask)
        input_segments_list.append(segment_ids)
        idx += step_size
        n_segment += 1

    all_input_ids = tf.concat(input_ids_list, axis=0)
    all_input_mask = tf.concat(input_masks_list, axis=0)
    all_segment_ids = tf.concat(input_segments_list, axis=0)
    print(all_input_ids)
    return all_input_ids, all_input_mask, all_segment_ids, n_segment


def model_fn_classification(model_config, train_config, model_class,
                            max_seq_length, query_len, total_doc_len,
                            special_flags=[], override_prediction_fn=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    tf_logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    query = features["query"]
    doc = features["doc"]
    doc_mask = features["doc_mask"]
    data_ids = features["data_id"]

    segment_len = max_seq_length - query_len - 3
    step_size = model_config.step_size
    input_ids, input_mask, segment_ids, n_segments = \
        iterate_over(query, doc, doc_mask, total_doc_len, segment_len, step_size)
    if mode == tf.estimator.ModeKeys.PREDICT:
        label_ids = tf.ones([input_ids.shape[0]], dtype=tf.int32)
    else:
        label_ids = features["label_ids"]
        label_ids = tf.reshape(label_ids, [-1])

    if "is_real_example" in features:
        is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
        is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if "feed_features" in special_flags:
        model = model_class(
            config=model_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            features=features,
        )
    else:
        model = model_class(
            config=model_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )
    if "new_pooling" in special_flags:
        pooled = mimic_pooling(model.get_sequence_output(), model_config.hidden_size, model_config.initializer_range)
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
            "output_weights", [train_config.num_classes, model_config.hidden_size],
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
                "logits": logits,
                "doc": doc,
                "data_ids": data_ids,
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

