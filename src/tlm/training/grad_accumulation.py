import numpy as np
import tensorflow as tf

from models.transformer import optimization_v2 as optimization


def get_accumulated_optimizer_from_config(loss, train_config, tvars, gradient_accmulation_multiplier):
    max_train_step = train_config.num_train_steps
    if train_config.no_lr_decay:
      max_train_step = 100000 * 100000

    train_op = get_accumulated_optimizer(
        loss,
        train_config.learning_rate,
        max_train_step,
        train_config.num_warmup_steps,
        train_config.use_tpu,
        tvars,
        gradient_accmulation_multiplier
    )
    return train_op

def get_accumulated_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu,
                              tvars, gradient_accmulation_multiplier):
    global_step = tf.compat.v1.train.get_or_create_global_step()

    learning_rate = optimization.get_learning_rate(global_step, init_lr, num_train_steps, num_warmup_steps)
    optimizer = optimization.AdamWeightDecayOptimizer(
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

    # compute batch gradient
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    # this is a list of sum(dy/dx) for each variable that must be paired with a tvars list.
    # element may be an IndexedSlices object that does not support assignning, e.g. [g.assign(value) for g in grads]
    # some of the elements are None, meaning y and x does not depend on each other.
    # Nonetypes must be handled using Python, tensorflow cannot convert Nonetypes to 0.

    # declare a temp variable for summation
    sum_gradient = list([ tf.Variable(name="sum_grads" + str(i),
                                 initial_value=np.zeros(tv.shape),
                                 trainable=False,
                                 dtype=tf.float32,
                                 ) for i, tv in enumerate(tvars)])
    sum_ops = []
    unused_variable_in_batch = []

    # gradient accumulation
    for i, gv in enumerate(grads):
        if gv is not None:
            sum_ops.append(sum_gradient[i].assign_add(gv, name="accumulate_gradient"))
        else:
            unused_variable_in_batch.append(sum_gradient[i])
            sum_gradient[i] = None

    # NOTE : calling .assign_add does NOTHING in estimator, must wrap them all and handle them via train_ops

    def apply_accumulated_gradients(sums):
        # normalize gradient
        normalize_ops = []
        for i, g in enumerate(sums):
            if g is not None:
                normalize_ops.append(sums[i].assign(tf.multiply(g, 1 / gradient_accmulation_multiplier)))
                # assign to make sure it still is a variable, or else it will become a Tensor
        with tf.control_dependencies(normalize_ops):
            minimize_op = optimizer.apply_gradients(zip(sums, tvars), global_step=global_step)
        return tf.group(minimize_op, *normalize_ops, name="apply_accumulated_gradients")

    train_op = tf.cond(tf.math.equal(global_step % gradient_accmulation_multiplier, 0),
                       lambda: apply_accumulated_gradients(sum_gradient),
                       lambda: optimizer.apply_gradients(zip([None for _ in grads], tvars), global_step=global_step))

    # reset accumulation when necessary
    def reset():
        counter = 0
        for i, s in enumerate(sum_gradient):
            if s is None:
                # restore reference from None to the original variable
                sum_gradient[i] = unused_variable_in_batch[counter]
                counter += 1
        return tf.group([s.assign(tf.zeros_like(s)) for s in sum_gradient])

    with tf.control_dependencies([train_op]):
        reset_ops = tf.cond(tf.math.equal(global_step % gradient_accmulation_multiplier, 0),
                            reset,
                            tf.no_op)
    # the 2 branches must have identical structure, [op1, op2, ...] || no_op cannot be valid cond branch.
    # tf.group to convert all resets into 1 op and match with no_op: tf.group() || np_op

    # Increment global step
    new_global_step = global_step + 1
    train_op = tf.group(*sum_ops, [train_op, global_step.assign(new_global_step), reset_ops])
    return train_op
