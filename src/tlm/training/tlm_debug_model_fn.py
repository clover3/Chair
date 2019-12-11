import tensorflow as tf

from tlm.tlm.tlm2_network import tlm2_raw_prob
from tlm.training.lm_model_fn import get_tlm_assignment_map_v2


def model_fn_tlm_debug(bert_config, train_config, model_config, logging, model_class):

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    logging.info("*** Features ***")
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    next_sentence_labels = features["next_sentence_labels"]
    tlm_prefix = "target_task"

    with tf.compat.v1.variable_scope(tlm_prefix):
      output, prob1, prob2 = tlm2_raw_prob(bert_config, train_config.use_one_hot_embeddings,
                                           input_ids, input_mask, segment_ids)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = model_class(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=train_config.use_one_hot_embeddings,
    )

    all_vars = tf.compat.v1.all_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    (assignment_map, assignment_map_tt, initialized_variable_names
      ) = get_tlm_assignment_map_v2(all_vars, tlm_prefix, train_config.init_checkpoint, train_config.second_init_checkpoint)

    def load_fn():
      if train_config.init_checkpoint:
        tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
      tf.compat.v1.train.init_from_checkpoint(train_config.second_init_checkpoint, assignment_map_tt)

    if train_config.use_tpu:
      def tpu_scaffold():
        load_fn()
        return tf.compat.v1.train.Scaffold()
      scaffold_fn = tpu_scaffold
    else:
      load_fn()

    tvars = [v for v in all_vars if not v.name.startswith(tlm_prefix)]

    logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "input_ids": input_ids,
          "prob1": prob1,
          "prob2": prob2,
          "scores": output,
      }
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=0,
          predictions=predictions,
          scaffold_fn=scaffold_fn)


    return output_spec

  return model_fn


