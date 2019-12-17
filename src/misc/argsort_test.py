import tensorflow as tf

input_ids = tf.ones([10,6])


rand = tf.random.uniform(
    input_ids.shape,
    minval=0,
    maxval=1,
    dtype=tf.dtypes.float32,
    seed=0,
    name=None
)
random_seq = tf.argsort(
    rand,
    axis=-1,
    direction='DESCENDING',
    stable=False,
    name=None
)

print(random_seq)
l = [random_seq[:,0:2],
     random_seq[:,2:4],
     random_seq[:,4:6]]

locations = tf.concat(l, axis=0)

grouped_positions = tf.transpose(tf.reshape(locations, [3,-1, 2]), [1,0,2])

print(grouped_positions)