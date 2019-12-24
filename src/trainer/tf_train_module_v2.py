import tensorflow as tf


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
