import tensorflow as tf
from tensorflow_ranking.python.losses import RankingLossKey, make_loss_fn

import tlm.training.assignment_map
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import dropout
from models.transformer.optimization_v2 import create_simple_optimizer
from tf_util.tf_logging import tf_logging
from tlm.model.base import mimic_pooling
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments, classification_metric_fn


def shift(v):
    return tf.math.floormod(v+2, 3)


def model_fn_ranking_ndcg(bert_config, train_config, model_class, special_flags=[]):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    tf_logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
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
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
            features=features,
        )
    else:
        model = model_class(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )
    if "new_pooling" in special_flags:
        pooled = mimic_pooling(model.get_sequence_output(), bert_config.hidden_size, bert_config.initializer_range)
    else:
        pooled = model.get_pooled_output()

    if train_config.checkpoint_type != "bert_nli" and train_config.use_old_logits:
        tf_logging.info("Use old version of logistic regression")
        logits = tf.keras.layers.Dense(train_config.num_classes, name="cls_dense")(pooled)
    else:
        tf_logging.info("Use fixed version of logistic regression")
        output_weights = tf.compat.v1.get_variable(
            "output_weights", [3, bert_config.hidden_size],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
        )

        output_bias = tf.compat.v1.get_variable(
            "output_bias", [3],
            initializer=tf.compat.v1.zeros_initializer()
        )

        if is_training:
            pooled = dropout(pooled, 0.1)

        logits = tf.matmul(pooled, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

    # TODO given topic_ids, reorder logits to [num_group, num_max_items]
    #
    ndcg_rankin_loss = make_loss_fn(RankingLossKey.APPROX_NDCG_LOSS)
    one_hot_labels = tf.one_hot(label_ids, 3)
    # logits ; [batch_size, num_classes]
    logits_t = tf.transpose(logits, [1,0])
    one_hot_labels_t = tf.transpose(one_hot_labels, [1,0])
    fake_logit = logits_t * 1e-5 + one_hot_labels_t
    loss_arr = ndcg_rankin_loss(fake_logit, one_hot_labels_t, {})
    loss = loss_arr
    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}

    if train_config.checkpoint_type == "bert":
        assignment_fn = tlm.training.assignment_map.get_bert_assignment_map
    elif train_config.checkpoint_type == "v2":
        assignment_fn = tlm.training.assignment_map.assignment_map_v2_to_v2
    elif train_config.checkpoint_type == "bert_nli":
        assignment_fn = tlm.training.assignment_map.get_bert_nli_assignment_map
    elif train_config.checkpoint_type == "attention_bert":
        assignment_fn = tlm.training.assignment_map.bert_assignment_only_attention
    elif train_config.checkpoint_type == "attention_bert_v2":
        assignment_fn = tlm.training.assignment_map.assignment_map_v2_to_v2_only_attention
    elif train_config.checkpoint_type == "wo_attention_bert":
        assignment_fn = tlm.training.assignment_map.bert_assignment_wo_attention
    elif train_config.checkpoint_type == "as_is":
        assignment_fn = tlm.training.assignment_map.get_assignment_map_as_is
    else:
        if not train_config.init_checkpoint:
            pass
        else:
            raise Exception("init_checkpoint exists, but checkpoint_type is not specified")

    scaffold_fn = None
    if train_config.init_checkpoint:
      assignment_map, initialized_variable_names = assignment_fn(tvars, train_config.init_checkpoint)

      def init_fn():
        tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
      scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
    log_var_assignments(tvars, initialized_variable_names)

    TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        if "simple_optimizer" in special_flags:
            tf_logging.info("using simple optimizer")
            train_op = create_simple_optimizer(loss, train_config.learning_rate, train_config.use_tpu)
        else:
            train_op = optimization.create_optimizer_from_config(loss, train_config)
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
        output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions=predictions,
                scaffold_fn=scaffold_fn)


    return output_spec

  return model_fn



