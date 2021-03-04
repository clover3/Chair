import tlm.training.assignment_map
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import dropout
from models.transformer.optimization_v2 import create_simple_optimizer
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.base import mimic_pooling
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2, triple_model_prefix1, \
    triple_model_prefix2, triple_model_prefix3
from tlm.training.assignment_map import get_init_fn_for_two_checkpoints, \
    get_init_fn_for_three_checkpoints, get_init_fn_for_phase1_load_and_bert, \
    get_init_fn_for_two_checkpoints_ex, get_init_fn_for_cppnc_start, get_init_fn_for_phase2_phase1_remap
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments, classification_metric_fn, \
    reweight_zero


def shift(v):
    return tf.math.floormod(v+2, 3)


def model_fn_classification(bert_config, train_config, model_class, special_flags=[], override_prediction_fn=None):
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


def get_init_fn(train_config, tvars):
    num_checkpoint = 1
    if train_config.checkpoint_type.startswith("two_checkpoints"):
        num_checkpoint = 2
    elif train_config.checkpoint_type == "three_checkpoints":
        num_checkpoint = 3

    if num_checkpoint == 1:
        assignment_fn = get_assignment_fn_from_checkpoint_type(train_config.checkpoint_type,
                                                               train_config.init_checkpoint)
        assignment_map, initialized_variable_names = assignment_fn(tvars, train_config.init_checkpoint)

        def init_fn():
            tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
        return initialized_variable_names, init_fn
    elif num_checkpoint == 2:
        if train_config.checkpoint_type == "two_checkpoints":
            return get_init_fn_for_two_checkpoints(train_config,
                                                   tvars,
                                                   train_config.init_checkpoint,
                                                   dual_model_prefix1,
                                                   train_config.second_init_checkpoint,
                                                   dual_model_prefix2)
        elif train_config.checkpoint_type == "two_checkpoints_cppnc_start":
            return get_init_fn_for_cppnc_start(train_config,
                                                       tvars,
                                                       train_config.init_checkpoint,
                                                       dual_model_prefix1,
                                                       train_config.second_init_checkpoint,
                                                       dual_model_prefix2)

        elif train_config.checkpoint_type == "two_checkpoints_phase2_to_phase1":
            return get_init_fn_for_phase2_phase1_remap(train_config,
                                                           tvars,
                                                           train_config.init_checkpoint,
                                                           dual_model_prefix1,
                                                           train_config.second_init_checkpoint,
                                                           dual_model_prefix2)
        elif train_config.checkpoint_type == "two_checkpoints_phase1_load_and_bert":
            return get_init_fn_for_phase1_load_and_bert(train_config,
                                                            tvars,
                                                            train_config.init_checkpoint,
                                                            dual_model_prefix1,
                                                            train_config.second_init_checkpoint,
                                                            dual_model_prefix2)
        else:
            if train_config.checkpoint_type == "two_checkpoints_v1_v1":
                first_from_v1 = True
                second_from_v1 = True
            elif train_config.checkpoint_type == "two_checkpoints_v1_v2":
                first_from_v1 = True
                second_from_v1 = False
            elif train_config.checkpoint_type == "two_checkpoints_v2_v1":
                first_from_v1 = False
                second_from_v1 = True
            elif train_config.checkpoint_type == "two_checkpoints_v2_v2":
                first_from_v1 = False
                second_from_v1 = False
            else:
                assert False

            return get_init_fn_for_two_checkpoints_ex(first_from_v1, second_from_v1,
                                                         tvars,
                                                         train_config.init_checkpoint,
                                                         dual_model_prefix1,
                                                         train_config.second_init_checkpoint,
                                                         dual_model_prefix2)

    elif num_checkpoint == 3:
        return get_init_fn_for_three_checkpoints(train_config,
                                               tvars,
                                               train_config.init_checkpoint,
                                               triple_model_prefix1,
                                               train_config.second_init_checkpoint,
                                               triple_model_prefix2,
                                               train_config.third_init_checkpoint,
                                               triple_model_prefix3,
                                                 )

    else:
        raise Exception("Unexpected num_checkpoint={}".format(num_checkpoint))


def get_assignment_fn_from_checkpoint_type(checkpoint_type, init_checkpoint):
    if checkpoint_type == "bert":
        assignment_fn = tlm.training.assignment_map.get_bert_assignment_map
    elif checkpoint_type == "v2":
        assignment_fn = tlm.training.assignment_map.assignment_map_v2_to_v2
    elif checkpoint_type == "bert_nli":
        assignment_fn = tlm.training.assignment_map.get_bert_nli_assignment_map
    elif checkpoint_type == "attention_bert":
        assignment_fn = tlm.training.assignment_map.bert_assignment_only_attention
    elif checkpoint_type == "attention_bert_v2":
        assignment_fn = tlm.training.assignment_map.assignment_map_v2_to_v2_only_attention
    elif checkpoint_type == "wo_attention_bert":
        assignment_fn = tlm.training.assignment_map.bert_assignment_wo_attention
    elif checkpoint_type == "as_is":
        assignment_fn = tlm.training.assignment_map.get_assignment_map_as_is
    elif checkpoint_type == "model_has_it":
        pass
    else:

        if not init_checkpoint:
            assignment_fn = None
        elif not checkpoint_type:
            raise Exception("init_checkpoint exists, but checkpoint_type is not specified")
        else:
            raise Exception("Unknown checkpoint_type : {}".format(checkpoint_type))
    return assignment_fn




