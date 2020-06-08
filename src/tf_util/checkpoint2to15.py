import re
import sys

import numpy as np
import tensorflow as tf

from misc.show_checkpoint_vars import load_checkpoint_vars


def work(model_path, save_path):
    model = load_checkpoint_vars(model_path)
    var_list = []
    with tf.Session() as sess:
        for key in model:
            new_name = key
            new_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", new_name)
            new_name = re.sub("dense[_]?\d*", "dense", new_name)
            new_name = re.sub("cls_dense/kernel", "output_weights", new_name)
            new_name = re.sub("cls_dense/bias", "output_bias", new_name)

            var_value = model[key]
            if new_name == "output_weights":
                print(var_value.shape)
                var_value = np.transpose(var_value, [1, 0])

            new_var = tf.Variable(var_value, name=new_name)
            var_list.append(new_var)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, save_path)


if __name__ == "__main__":
    work(sys.argv[1], sys.argv[2])
