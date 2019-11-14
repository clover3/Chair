import tensorflow as tf

def init_session():
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)



def get_train_op(loss, lr, name='Adam'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8,
                                       name=name)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, global_step
