import tensorflow as tf


@tf.custom_gradient
def categorical_sampling(x):
    idx = tf.random.categorical(x, 1)

    def grad(dy):
        return tf.expand_dims(dy, 1) * x

    return idx, grad


@tf.custom_gradient
def gather(param, indice):
    y = tf.gather_nd(param, indice)

    def grad(dy):
        g1 = tf.scatter_nd(updates=dy, indices=indice, shape=param.shape)
        return g1, dy
    return y, grad


def code():
    # [batch, data_size]
    x = tf.random.uniform([10, 4],0,1)
    z = tf.constant([1,2,3,4], dtype=tf.float32)
    with tf.GradientTape(True) as g:
        g.watch(x)
        g.watch(z)
        idx = categorical_sampling(x)

        print(idx)
        y_shape = idx.shape[:-1] + z.shape[idx.shape[-1]:]
        y = gather(z, idx)

    print("x:", x)
    print("Y:", y)
    dy_dz = g.gradient(y, z)
    dy_dx = g.gradient(y, x)
    print("Gradient dy_dz: ", dy_dz)
    print("Gradient dy_dx: ", dy_dx)


def code_13():
    #tf.compat.v1.disable_eager_execution()

    z = tf.constant([1.,2,3,4])
    w = tf.Variable([1.,2.])

    y = z * w[1:]
    ygold = tf.constant([1.,2,3,4])

    loss = tf.keras.losses.MAE(ygold, y)

    y = z[:3]
    g = tf.gradients(y[0], z)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.98, epsilon=1e-8,
                                       )
    train_op = optimizer.minimize(loss, global_step=global_step)

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False
                            )
    sess = tf.compat.v1.Session(config=config)
    sess.run(tf.compat.v1.global_variables_initializer())
    print(y)
    print(g)

    for i in range(10):
        y_,w_, _ = sess.run([y,w, train_op])
        print("y_:", y_)
        print("w_", w_)

