import tensorflow as tf
from models.transformer import bert_common_v2 as bert_common
from models.transformer import optimization_v2 as optimization
from trainer.get_param_num import get_param_num
from tlm.model.lm_objective import get_masked_lm_output, get_next_sentence_output
from data_generator.special_tokens import MASK_ID, PAD_ID
from tlm.model.masking import random_masking, biased_masking
import tlm.training.target_mask_hp as target_mask_hp
from tlm.model.nli_ex_v2 import transformer_nli
from models.transformer import hyperparams
import collections
import re
from tlm.tf_logging import tf_logging

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

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.random.set_seed(0)
        seed = 0
        print("Seed as zero")
    else:
        seed = None

    masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = random_masking(input_ids,
                                                                                         input_mask,
                                                                                         train_config.max_predictions_per_seq,
                                                                                         MASK_ID, seed)

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

    total_loss = masked_lm_loss

    tvars = tf.compat.v1.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if train_config.init_checkpoint:
      if train_config.checkpoint_type == "":
        assignment_map, initialized_variable_names = get_bert_assignment_map(tvars, train_config.init_checkpoint)
        if train_config.use_tpu:
          def tpu_scaffold():
            tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
            return tf.compat.v1.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
      elif train_config.checkpoint_type == "nli_and_bert":
        assignment_map, initialized_variable_names = get_bert_assignment_map(tvars, train_config.init_checkpoint)
        assignment_map2, initialized_variable_names2 = get_cls_assignment(tvars, train_config.second_init_checkpoint)
        for vname in initialized_variable_names2:
          logging.info("Loading from 2nd checkpoint : %s" % vname)
        for k,v in initialized_variable_names2.items():
          initialized_variable_names[k] = v
        def init_fn():
          tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
          tf.compat.v1.train.init_from_checkpoint(train_config.second_init_checkpoint, assignment_map2)
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
          "masked_input_ids":masked_input_ids,
          "masked_lm_example_loss":masked_lm_example_loss,
          "masked_lm_positions":masked_lm_positions,
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

    step = 200
    pad_len = 200 - 1 - (512 - (step * 2 - 1))
    def spread(t):
        cls_token = t[:,:1]
        pad = tf.ones([batch_size, pad_len], tf.dtypes.int32) * PAD_ID
        a = t[:, :step]
        b = tf.concat([cls_token, t[:,step:step*2-1]], axis=1)
        c = tf.concat([cls_token, t[:,step*2-1:], pad], axis=1)
        return tf.concat([a,b,c], axis=0)

    def collect(t):
        a = t[:batch_size]
        b = t[batch_size:batch_size *2, 1:]
        c = t[batch_size*2:, 1:-pad_len]
        return tf.concat([a,b,c], axis=1)

    model = transformer_nli(hp, spread(input_ids), spread(input_mask), spread(segment_ids), voca_size, method, False)
    output = model.conf_logits
    output = collect(output)
    return output


tlm_prefix = "target_task"

def get_bert_assignment_map(tvars, lm_checkpoint):
    lm_assignment_candidate = {}
    real_name_map = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
        targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
        lm_assignment_candidate[targ_name] = var
        tf_logging.info("Init from lm_checkpoint : %s" % name)
        real_name_map[targ_name] = name

    assignment_map = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in lm_assignment_candidate:
                continue
            assignment_map[name] = lm_assignment_candidate[name]

            tvar_name = real_name_map[name]

            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


def get_cls_assignment(tvars, lm_checkpoint):
    lm_assignment_candidate = {}
    real_name_map = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        tokens = name.split("/")
        top_scope = tokens[0]
        targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
        targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
        lm_assignment_candidate[targ_name] = var
        real_name_map[targ_name] = name

    assignment_map = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in lm_assignment_candidate:
                continue
            if not name.startswith("cls"):
                continue
            assignment_map[name] = lm_assignment_candidate[name]

            tvar_name = real_name_map[name]

            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return (assignment_map, initialized_variable_names)



def get_tlm_assignment_map(tvars, lm_checkpoint, target_task_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}
    real_name_map = {}

    target_task_name_to_var = collections.OrderedDict()
    lm_assignment_candidate = {}
    tt_assignment_candidate = {}
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
            targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", inner_name)
            targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
            tt_assignment_candidate[targ_name] = var
            tf_logging.info("Init from target_task_checkpoint : %s" % name)
        else:
            targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
            targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
            lm_assignment_candidate[targ_name] = var
            tf_logging.info("Init from lm_checkpoint : %s" % name)

        real_name_map[targ_name] = name

    assignment_map_tt = collections.OrderedDict()
    if target_task_checkpoint:
        for x in tf.train.list_variables(target_task_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in tt_assignment_candidate:
                continue
            assignment_map_tt[name] = tt_assignment_candidate[name]

            real_name = real_name_map[name]
            initialized_variable_names[real_name] = 1

    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in lm_assignment_candidate:
                continue
            assignment_map[name] = lm_assignment_candidate[name]
            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1

    return assignment_map, assignment_map_tt, initialized_variable_names


def get_assignment_map_as_is(tvars, checkpoint):
    current_vars = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        current_vars[name] = var
        tf_logging.info("Init from lm_checkpoint : %s" % name)

    assignment_map = {}
    initialized_variable_names = {}
    if checkpoint:
        for x in tf.train.list_variables(checkpoint):
            (name, var) = (x[0], x[1])
            if name not in current_vars:
                continue
            assignment_map[name] = current_vars[name]

            initialized_variable_names[name] = 1
            initialized_variable_names[name + ":0"] = 1

    return assignment_map, initialized_variable_names


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

    priority_model = get_nli_ex_model_segmented
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

    all_vars = tf.compat.v1.all_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    (assignment_map, assignment_map_tt, initialized_variable_names
      ) = get_tlm_assignment_map(all_vars, train_config.init_checkpoint, train_config.second_init_checkpoint)

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



