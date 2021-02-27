import sys

import tensorflow as tf

from misc.show_checkpoint_vars import load_checkpoint_vars
from tlm.model.dual_model_common import dual_model_prefix2


def parse_shift_name(name, key, shift_idx):
    tokens = name.split("/")
    new_tokens = []
    for token in tokens:
        if token.startswith(key):
            if token == key:
                idx = 0
            else:
                idx_str = token[len(key)+1:]
                idx = int(idx_str)
            new_idx = idx + shift_idx

            new_token = "_".join([key, str(new_idx)])
        else:
            new_token = token

        new_tokens.append(new_token)
    return "/".join(new_tokens)


def work(model_path, save_path):
    model = load_checkpoint_vars(model_path)
    var_list = []
    source_prefix = dual_model_prefix2
    with tf.compat.v1.Session() as sess:
        for old_name in model:
            if old_name.startswith(source_prefix):
                drop_n = len(source_prefix) + 1
                new_name = old_name[drop_n:]
                if "/dense" in new_name:
                    new_name = parse_shift_name(new_name, "dense", -37)
                if "/layer_normalization" in new_name:
                    new_name = parse_shift_name(new_name, "layer_normalization", -25)
                var_value = model[old_name]
                new_var = tf.Variable(var_value, name=new_name)
                var_list.append(new_var)
                print("Old: " + old_name)
                print("New: " + new_name)
            elif old_name.startswith("cls_dense_1/"):
                var_value = model[old_name]
                new_name = old_name.replace("cls_dense_1/", "cls_dense/")
                new_var = tf.Variable(var_value, name=new_name)
                var_list.append(new_var)
                print("Old: " + old_name)
                print("New: " + new_name)

            else:
                pass

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.save(sess, save_path)


if __name__ == "__main__":
    work(sys.argv[1], sys.argv[2])
