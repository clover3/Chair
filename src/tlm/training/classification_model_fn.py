import tensorflow as tf

from models.transformer import optimization_v2 as optimization
from tlm.training import lm_model_fn
from trainer.get_param_num import get_param_num


def shift(v):
    return tf.math.floormod(v+2, 3)


def model_fn_classification(bert_config, train_config, logging, model_class):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    logging.info("*** Features ***")
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    label_ids = tf.reshape(label_ids, [-1])

    if "is_real_example" in features:
        is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
        is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = model_class(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=train_config.use_one_hot_embeddings,
    )

    pooled = model.get_pooled_output()
    if train_config.checkpoint_type != "bert_nli":
        logits = tf.keras.layers.Dense(train_config.num_classes, name="cls_dense")(pooled)
    else:
        output_weights = tf.compat.v1.get_variable(
            "output_weights", [3, bert_config.hidden_size],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
        )

        output_bias = tf.compat.v1.get_variable(
            "output_bias", [3],
            initializer=tf.zeros_initializer()
        )

        if is_training:
            pooled = tf.layers.dropout(pooled,
                                     rate=0.1,
                                     training=tf.convert_to_tensor(is_training))

        logits = tf.matmul(pooled, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

    print('label_ids', label_ids.shape)

    loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=label_ids)
    loss = tf.reduce_mean(input_tensor=loss_arr)
    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}

    if train_config.checkpoint_type == "bert":
        assignment_fn = lm_model_fn.get_bert_assignment_map
    elif train_config.checkpoint_type == "v2":
        assignment_fn = lm_model_fn.assignment_map_v2_to_v2
    elif train_config.checkpoint_type == "bert_nli":
        assignment_fn = lm_model_fn.get_bert_nli_assignment_map
    else:
        raise Exception("checkpoint_type not specified")

    scaffold_fn = None
    if train_config.init_checkpoint:
      assignment_map, initialized_variable_names = assignment_fn(tvars, train_config.init_checkpoint)
      if train_config.use_tpu:

        def tpu_scaffold():
          tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
          return tf.compat.v1.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)

    logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    logging.info("Total parameters : %d" % get_param_num())

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer_from_config(loss, train_config)
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
      )
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(log_probs, label, is_real_example):
          """Computes the loss and accuracy of the model."""
          log_probs = tf.reshape(
              log_probs, [-1, log_probs.shape[-1]])
          pred = tf.argmax(
              input=log_probs, axis=-1, output_type=tf.int32)

          label = tf.reshape(label, [-1])
          accuracy = tf.compat.v1.metrics.accuracy(
              labels=label, predictions=pred, weights=is_real_example)

          return {
              "accuracy": accuracy,
          }

      eval_metrics = (metric_fn, [
          logits, label_ids, is_real_example
      ])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)

    return output_spec

  return model_fn