import sys

import tensorflow as tf

from misc.show_checkpoint_vars import load_checkpoint_vars


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


def rename_and_save(input_checkpoint_path, save_path):
    source_ckpt = load_checkpoint_vars(input_checkpoint_path)
    var_list = []
    name_set = set()
    with tf.compat.v1.Session() as sess:
        for old_name in source_ckpt:
            tokens = old_name.split("/")
            if tokens[0] == "SCOPE1":
                new_name = "/".join(tokens[1:])
            else:
                new_name = old_name
            new_var = tf.Variable(source_ckpt[old_name], name=new_name)
            var_list.append(new_var)
            assert new_name not in name_set
            print(old_name, new_name)
            name_set.add(new_name)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.save(sess, save_path)


if __name__ == "__main__":
    rename_and_save(sys.argv[1], sys.argv[2])