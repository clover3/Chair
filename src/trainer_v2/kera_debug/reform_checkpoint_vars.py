import numpy as np

from misc.show_checkpoint_vars import load_checkpoint_vars
from my_tf import tf


def reshape_and_save(input_checkpoint_path, save_path):
    source_ckpt = load_checkpoint_vars(input_checkpoint_path)
    var_list = []
    with tf.compat.v1.Session() as sess:
        for name in source_ckpt:
            var = source_ckpt[name]
            new_shape = None
            for role in ["query", "key", "value"]:
                if name.endswith(f"attention/self/{role}/kernel"):
                    new_shape = [768, 12, 64]
                if name.endswith(f"attention/self/{role}/bias"):
                    new_shape = [12, 64]
            if name.endswith("attention/output/dense/kernel"):
                new_shape = [12, 64, 768]

            if new_shape is not None:
                print("{}: {}->{}".format(name, var.shape, new_shape))
                var = np.reshape(var, new_shape)

            new_var = tf.Variable(var, name=name)
            var_list.append(new_var)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.save(sess, save_path)


if __name__ == "__main__":
    src_path = "C:\\work\\Code\\Chair\\output\\model\\runs\\uncased_L-12_H-768_A-12\\bert_model.ckpt"
    out_path = "C:\\work\\Code\\Chair\\output\\model\\runs\\reshape_bert\\uncased_L-12_H-768_A-12\\bert_model.ckpt"
    reshape_and_save(src_path, out_path)