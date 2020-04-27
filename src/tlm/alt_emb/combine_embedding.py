import tensorflow as tf

from misc.show_checkpoint_vars import load_checkpoint_vars


def combine(nli_checkpoint, alt_emb_checkpoint, save_path):
    print("Combining...")
    # Write variables names of nli_checkpoint on alt_emb_checkpoint
    nli_d = load_checkpoint_vars(nli_checkpoint)
    alt_emb_d = load_checkpoint_vars(alt_emb_checkpoint)
    var_list = []
    with tf.Session() as sess:
        for key in alt_emb_d:
            if key in nli_d:
                new_var = tf.Variable(nli_d[key], name=key)
            else:
                new_var = tf.Variable(alt_emb_d[key], name=key)

            var_list.append(new_var)
        for key in nli_d:
            if key not in alt_emb_d:
                new_var = tf.Variable(nli_d[key], name=key)
                var_list.append(new_var)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, save_path)
