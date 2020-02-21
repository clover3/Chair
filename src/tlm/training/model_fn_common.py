import tensorflow as tf

from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import dropout
from tf_util.tf_logging import tf_logging
from trainer.get_param_num import get_param_num


def get_tpu_scaffold_or_init(init_fn, use_tpu):
    if use_tpu:
        def tpu_scaffold():
            init_fn()
            return tf.compat.v1.train.Scaffold()

        scaffold_fn = tpu_scaffold
        return scaffold_fn
    else:
        init_fn()
        return None

def log_var_assignments_one_by_one(tvars, initialized_variable_names, initialized_variable_names2=None):
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        if initialized_variable_names2 is not None:
            if var.name in initialized_variable_names2:
                init_string = ", *INIT_FROM_CKPT2*"
        if init_string:
            tf_logging.debug("    name = %s, shape = %s%s", var.name, var.shape,
                             init_string)
        else:
            tf_logging.info("    name = %s, shape = %s%s", var.name, var.shape,
                            " - Not Initialized")


def log_var_assignments(tvars, initialized_variable_names, initialized_variable_names2=None):
    tf_logging.info("**** Trainable Variables ****")

    num_init_vars = len(initialized_variable_names)
    if initialized_variable_names2 is not None:
        num_init_vars += len(initialized_variable_names2)

    if num_init_vars == len(tvars):
        tf_logging.info("All variables initialized")
    elif num_init_vars == 0:
        tf_logging.info("No variables initialized")
    else:
        log_var_assignments_one_by_one(tvars, initialized_variable_names, initialized_variable_names2)
    tf_logging.info("Total parameters : %d" % get_param_num())


def align_checkpoint(tvars, init_checkpoint, assignment_fn):
    if init_checkpoint:
        assignment_map, initialized_variable_names = assignment_fn(tvars, init_checkpoint)

        def init_fn():
            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    else:
        initialized_variable_names = {}

        def init_fn():
            pass

    return initialized_variable_names, init_fn


def align_checkpoint_twice(tvars, init_checkpoint, assignment_fn):
    if init_checkpoint:
        map1, map2, init_vars = assignment_fn(tvars, init_checkpoint)

        def init_fn():
            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, map1)
            tf.compat.v1.train.init_from_checkpoint(init_checkpoint, map2)

    else:
        initialized_variable_names = {}

        def init_fn():
            pass

    return init_vars, init_fn


def get_training_spec(loss, mode, train_config, scaffold_fn):
    train_op = optimization.create_optimizer_from_config(loss, train_config)
    output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      scaffold_fn=scaffold_fn,
    )
    return output_spec


def reweight_zero(label_ids, loss_arr):
    is_zero = tf.cast(tf.equal(label_ids, 0), tf.float32)
    weight = tf.ones_like(loss_arr) - is_zero * 0.75
    loss_arr = loss_arr * weight * 3
    return loss_arr


class Classification:
    def __init__(self, num_classes, features, rep, is_training, loss_weighting=None):
        self.num_classes = num_classes
        self.label_ids = features["label_ids"]
        self.label_ids = tf.reshape(self.label_ids, [-1])
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(self.label_ids), dtype=tf.float32)
        self.is_real_example = is_real_example

        if is_training:
            rep = dropout(rep, 0.1)
        logits = tf.keras.layers.Dense(self.num_classes, name="cls_dense")(rep)

        self.logits = logits
        self.loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=self.label_ids)
        if loss_weighting is not None:
            print("Special flags : ", "bias_loss")
            self.loss_arr = loss_weighting(self.label_ids, self.loss_arr)

        self.loss = tf.reduce_mean(input_tensor=self.loss_arr)
        self.preds = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32)

    def eval_metrics(self):
        eval_metrics = (classification_metric_fn, [
            self.logits, self.label_ids, self.is_real_example
        ])
        return eval_metrics


def log_features(features):
    tf_logging.info("*** Features ***")
    for name in sorted(features.keys()):
        tf_logging.info("    name = %s, shape = %s" % (name, features[name].shape))


def classification_metric_fn(log_probs, label, is_real_example):
    """Computes the loss and accuracy of the model."""
    log_probs = tf.reshape(log_probs, [-1, log_probs.shape[-1]])
    pred = tf.argmax(
      input=log_probs, axis=-1, output_type=tf.int32)

    label = tf.reshape(label, [-1])
    accuracy = tf.compat.v1.metrics.accuracy(
      labels=label, predictions=pred, weights=is_real_example)

    return {
      "accuracy": accuracy,
    }


