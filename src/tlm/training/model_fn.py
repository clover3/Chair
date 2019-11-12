import tensorflow as tf
from models.transformer import bert_common_v2 as bert_common
from models.transformer import optimization_v2 as optimization
from trainer.get_param_num import get_param_num
from tlm.model.lm_objective import get_masked_lm_output, get_next_sentence_output
from data_generator.special_tokens import MASK_ID
from tlm.model.masking import random_masking, biased_masking
import tlm.training.target_mask_hp as target_mask_hp
from tlm.model.nli_ex_v2 import transformer_nli
from models.transformer import hyperparams
import collections
import re

def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, next_sentence_example_loss,
              next_sentence_log_probs, next_sentence_labels):
    """Computes the loss and accuracy of the model."""
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(
        input=masked_lm_log_probs, axis=-1, output_type=tf.int32)
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)
    masked_lm_mean_loss = tf.compat.v1.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights)

    next_sentence_log_probs = tf.reshape(
        next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
    next_sentence_predictions = tf.argmax(
        input=next_sentence_log_probs, axis=-1, output_type=tf.int32)
    next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
    next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
        labels=next_sentence_labels, predictions=next_sentence_predictions)
    next_sentence_mean_loss = tf.compat.v1.metrics.mean(
        values=next_sentence_example_loss)

    return {
        "masked_lm_accuracy": masked_lm_accuracy,
        "masked_lm_loss": masked_lm_mean_loss,
        "next_sentence_accuracy": next_sentence_accuracy,
        "next_sentence_loss": next_sentence_mean_loss,
    }



def model_fn_random_masking(bert_config, train_config, logging, model_class):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    logging.info("*** Features ***")
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    next_sentence_labels = features["next_sentence_labels"]

    masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = random_masking(input_ids,
                                                                                         input_mask,
                                                                                         train_config.max_predictions_per_seq,
                                                                                         MASK_ID)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = model_class(
        config=bert_config,
        is_training=is_training,
        input_ids=masked_input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=train_config.use_one_hot_embeddings,
    )

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if train_config.init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = bert_common.get_assignment_map_from_checkpoint(tvars, train_config.init_checkpoint)
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
      train_op = optimization.create_optimizer_from_config(total_loss, train_config)
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      predictions = {
          "input_ids":input_ids,
      }
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          predictions=predictions,
          scaffold_fn=scaffold_fn)


    return output_spec

  return model_fn


def get_nli_ex_model(input_ids, input_mask, segment_ids):
  method = 5
  hp = hyperparams.HPBert()
  voca_size = 30522

  model = transformer_nli(hp, input_ids, input_mask, segment_ids, voca_size, method, False)
  output = model.conf_logits

  return output


def get_nli_ex_model_segmented(input_ids, input_mask, segment_ids):
    method = 5
    hp = hyperparams.HPBert()
    voca_size = 30522
    sequence_shape = bert_common.get_shape_list2(input_ids)
    batch_size = sequence_shape[0]

    def spread(t):
        return tf.concat([t[:,:200], t[:,200:400], t[:,400:]], 0)

    def collect(t):
        return tf.concat([t[:batch_size] , t[batch_size:batch_size *2], t[batch_size *2:]], 1)

    model = transformer_nli(hp, spread(input_ids), spread(input_mask), spread(segment_ids), voca_size, method, False)
    output = model.conf_logits
    output = collect(output)
    return output


tlm_prefix = "target_task"

def get_tlm_assignment_map(tvars, lm_checkpoint, target_task_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    target_task_name_to_var = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        if tlm_prefix == top_scope:
            inner_name = "/".join(tokens[1:])
            target_task_name_to_var[inner_name] = var
        name_to_variable[name] = var


    assignment_map = collections.OrderedDict()
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in name_to_variable:
                continue
            assignment_map[name] = name
            print(name)
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1

    assignment_map['/'] = tlm_prefix+"/"
    return (assignment_map, initialized_variable_names)




def model_fn_target_masking(bert_config, train_config, logging, model_class):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""
    logging.info("*** Features ***")
    for name in sorted(features.keys()):
      logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    next_sentence_labels = features["next_sentence_labels"]

    priority_model = get_nli_ex_model
    with tf.compat.v1.variable_scope(tlm_prefix):
      priority_score = tf.stop_gradient(priority_model(input_ids,
                                     input_mask,
                                     segment_ids))

    masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = biased_masking(input_ids,
                                                                                             input_mask,
                                                                                             priority_score,
                                                                                             target_mask_hp.alpha,
                                                                                             train_config.max_predictions_per_seq,
                                                                                             MASK_ID)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = model_class(
        config=bert_config,
        is_training=is_training,
        input_ids=masked_input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=train_config.use_one_hot_embeddings,
    )

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = masked_lm_loss + next_sentence_loss

    all_vars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    (assignment_map, initialized_variable_names
      ) = get_tlm_assignment_map(all_vars, train_config.init_checkpoint, train_config.target_task_checkpoint)

    def load_fn():
      if train_config.init_checkpoint:
        tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
      tf.compat.v1.train.init_from_checkpoint(train_config.target_task_checkpoint, assignment_map)

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
    logging.info("Total parameters : %d" % get_param_num())

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer_from_config(total_loss, train_config, tvars)
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      predictions = {
          "input_ids":input_ids,
      }
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          predictions=predictions,
          scaffold_fn=scaffold_fn)


    return output_spec

  return model_fn