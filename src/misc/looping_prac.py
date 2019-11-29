import tensorflow as tf



def code():
    # [batch, data_size]
    input_A = tf.zeros([10, 4])

    # [batch, n_item, data_size]
    input_b = tf.ones([50, 4])

    ab = [0]* 8 + [1] * 6 + [2] * 2 + [3] * 3 + [4] * 4 + \
        [5] * 5  + [6] * 6 + [7] * 7 + [8] * 8 + [9] * 1

    input_ba_map = tf.constant(ab)
    key = tf.gather(input_A, input_ba_map) #

    def encode(a,b):
        return a+b

    result_val = encode(input_b, key) # 50, 4
    score = result_val[:, 0] # 50
    indice = tf.stack([tf.range(50), input_ba_map], 1)
    collect_bin = tf.scatter_nd(indice, tf.ones([50]), [50,10])
    scattered_score = tf.transpose(tf.expand_dims(score,1) * collect_bin)

    best_match_idx = tf.argmax(scattered_score, axis=1)
    print(best_match_idx)

    b = tf.gather(input_b, best_match_idx)
    print(b)


code()
