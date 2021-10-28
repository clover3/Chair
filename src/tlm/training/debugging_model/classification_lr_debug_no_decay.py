from models.transformer.bert_common_v2 import dropout
from models.transformer.optimization_v2 import AdamWeightDecayOptimizer
from my_tf import tf
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import get_tpu_scaffold_or_init, log_var_assignments, classification_metric_fn


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu, tvars=None):
    """Creates an optimizer training op."""
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = is_warmup * warmup_learning_rate

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    if use_tpu:
        optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)

    if tvars is None:
        tvars = tf.compat.v1.trainable_variables()

    grads = tf.gradients(ys=loss, xs=tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


def create_optimizer_from_config(loss, train_config, tvars):
    max_train_step = train_config.num_train_steps
    if train_config.no_lr_decay:
        max_train_step = 100000 * 100000
    train_op = create_optimizer(
        loss,
        train_config.learning_rate,
        max_train_step,
        train_config.num_warmup_steps,
        train_config.use_tpu,
        tvars
    )
    return train_op


def model_fn_classification(model_config, train_config):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("  name = %s, shape = %s" % (name, features[name].shape))
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

        model_1 = BertModel(
            config=model_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )
        pooled = model_1.get_pooled_output()
        if is_training:
            pooled = dropout(pooled, 0.1)
        logits = tf.keras.layers.Dense(train_config.num_classes, name="cls_dense")(pooled)
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)
        loss = tf.reduce_mean(input_tensor=loss_arr)
        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, tvars)
            scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec

        global_step = tf.compat.v1.train.get_or_create_global_step()
        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            tvars = None
            train_op = create_optimizer_from_config(loss, train_config, tvars)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (classification_metric_fn, [
                logits, label_ids, is_real_example
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "input_ids": input_ids,
                    "label_ids": label_ids,
                    "logits": logits,
            }
            if "data_id" in features:
                predictions['data_id'] = features['data_id']
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)
        return output_spec
    return model_fn
