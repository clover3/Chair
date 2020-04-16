import tensorflow as tf



def load_checkpoint_vars(checkpoint_path):
    ckpt_reader = tf.train.load_checkpoint(checkpoint_path)

    d = {}
    for x in tf.train.list_variables(checkpoint_path):
        (name, var) = (x[0], x[1])
        d[name] = ckpt_reader.get_tensor(name)
    return d
