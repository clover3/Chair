import tensorflow as tf

def func():
    embedding_output_1 = tf.random.normal([2, 10, 5])
    embedding_output_2 = tf.random.normal([2, 10, 5])
    print(embedding_output_1)
    print(embedding_output_2)

    alt_emb_mask = tf.constant([[0,0,0,0,0, 1, 1, 0,0,0],[0,0,0,0,0, 1, 1, 0,0,0]])
    mask = tf.cast(tf.expand_dims(alt_emb_mask, 2), tf.float32)
    mask_neg = tf.cast(1 - tf.expand_dims(alt_emb_mask, 2), tf.float32)
    print(mask)
    print(mask_neg)

    r = embedding_output_1 * mask_neg + embedding_output_2 * mask
    print(r)

func()