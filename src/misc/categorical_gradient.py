import tensorflow as tf



def code():
    # [batch, data_size]
    x = tf.random.uniform([10, 4],0,1)
    z = tf.constant([1,2,3,4])
    with tf.GradientTape() as g:
        g.watch(x)
        g.watch(z)
        idx = tf.random.categorical(x, 1)
        idx = tf.argmax(x,1)
        y = tf.gather(z, idx)
    print(x)
    print(y)
    dy_dz = g.gradient(y, z)

    print("Gradient : ", dy_dz)


code()
