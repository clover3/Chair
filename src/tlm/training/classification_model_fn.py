import tensorflow as tf

from models.transformer import bert_common_v2 as bert_common
from models.transformer import optimization_v2 as optimization
from trainer.get_param_num import get_param_num


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

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = model_class(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=train_config.use_one_hot_embeddings,
    )

    enc = model.get_sequence_output()
    logits = tf.keras.layers.Dense(train_config.num_classes, name="cls_dense")(enc[:, 0, :])

    print('label_ids', label_ids.shape)

    loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=label_ids)
    loss = tf.reduce_mean(input_tensor=loss_arr)
    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if train_config.init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = bert_common.get_assignment_map_from_checkpoint(tvars, train_config.init_checkpoint)
      #assignment_map = bert_common.compress_assignment_map(assignment_map)
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
      def metric_fn(log_probs, label):
          """Computes the loss and accuracy of the model."""
          log_probs = tf.reshape(
              log_probs, [-1, log_probs.shape[-1]])
          pred = tf.argmax(
              input=log_probs, axis=-1, output_type=tf.int32)
          label = tf.reshape(label, [-1])
          accuracy = tf.compat.v1.metrics.accuracy(
              labels=label, predictions=pred)

          return {
              "accuracy": accuracy,
          }

      eval_metrics = (metric_fn, [
          logits, label_ids
      ])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)

    return output_spec

  return model_fn