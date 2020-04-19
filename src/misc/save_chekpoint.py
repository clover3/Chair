import tensorflow as tf

with tf.Session() as sess:
    val_list = []
    for key in alt_emb_d:
        if key in nli_d:
            print("nli_d", key)
            new_var = tf.Variable(nli_d[key], name=key)
        else:
            print("alt_emb_d", key)
            new_var = tf.Variable(alt_emb_d[key], name=key)

        var_list.append(new_var)
    for key in nli_d:
        if key not in alt_emb_d:
            new_var = tf.Variable(nli_d[key], name=key)
            var_list.append(new_var)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, save_path)

