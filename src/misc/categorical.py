import tensorflow as tf
from models.transformer.bert_common_v2 import get_shape_list, gather_index2d, get_shape_list2




def cate():
    n_sample = 3
    alpha = tf.constant(0.5)
    prob = tf.math.log(tf.constant([[0.5, 0.5, 0.01, 0.3, 0.2], [0.5, 0.5, 0.1, 0.03, 0.2]]))
    prob = tf.nn.softmax(prob, axis=1)
    sequence_shape = get_shape_list2(prob)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]

    rand = tf.random.uniform(
        prob.shape,
        minval=0,
        maxval=1,
        dtype=tf.dtypes.float32,
        seed=None,
        name=None
    )

    p1 = tf.ones_like(prob, dtype=tf.float32) / seq_length * alpha
    p2 = prob * (1-alpha)

    final_p = p1 + p2
    print(prob)
    print(final_p)

    _, indice = tf.math.top_k(
        rand * final_p,
        k=n_sample,
        sorted=False,
        name=None
    )

    print(indice)


cate()
