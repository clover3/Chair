import tensorflow as tf
import tensorflow.compat.v1 as tf1

from models.transformer.optimization_v2 import AdamWeightDecayOptimizer


def init_session():
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True

    return tf.compat.v1.Session(config=config)


def get_train_op(loss, lr, global_step = None, name='Adam',):
    if global_step is None:
        global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8,
                                       name=name)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, global_step


class OomReportingHook(tf.estimator.SessionRunHook):
    def before_run(self, run_context):
        return tf.estimator.SessionRunArgs(fetches=[],  # no extra fetches
                      options=tf.compat.v1.RunOptions(
                      report_tensor_allocations_upon_oom=True))




def get_train_op2(loss, lr, name='Adam', num_train_steps=0):
    global_step = tf1.train.get_or_create_global_step()

    learning_rate = get_learning_rate(global_step, lr, num_train_steps)

    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.02,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    tvars = tf1.trainable_variables()

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


def get_learning_rate(global_step, lr, num_train_steps):
    if num_train_steps:
        learning_rate = tf.constant(value=lr, shape=[], dtype=tf.float32)

        # Implements linear decay of the learning rate.
        learning_rate = tf.compat.v1.train.polynomial_decay(
            learning_rate,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)
    else:
        learning_rate = lr
    return learning_rate