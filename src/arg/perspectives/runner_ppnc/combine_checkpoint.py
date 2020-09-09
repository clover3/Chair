import sys

import tensorflow as tf

from misc.show_checkpoint_vars import load_checkpoint_vars
from tlm.model.dual_model_common import triple_model_prefix1, dual_model_prefix1, triple_model_prefix2, \
    dual_model_prefix2, triple_model_prefix3


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


def combine_pdcd_pc_bert(pc_bert_checkpoint, pdcd_checkpoint, save_path):
    pc_bert = load_checkpoint_vars(pc_bert_checkpoint)
    pdcd = load_checkpoint_vars(pdcd_checkpoint)
    var_list = []
    name_set = set()
    with tf.compat.v1.Session() as sess:
        for old_name in pc_bert:
            new_name = "{}/".format(triple_model_prefix1) + old_name
            new_var = tf.Variable(pc_bert[old_name], name=new_name)
            var_list.append(new_var)
            assert new_name not in name_set
            print(old_name, new_name)
            name_set.add(new_name)

        for old_name in pdcd:
            if dual_model_prefix1 in old_name:
                new_name = old_name.replace(dual_model_prefix1, triple_model_prefix2)
            elif dual_model_prefix2 in old_name:
                new_name = old_name.replace(dual_model_prefix2, triple_model_prefix3)
            else:
                new_name = old_name

            if "/dense" in new_name:
                parse_shift_name(new_name, "dense", 37)
            elif "/layer_normalization" in new_name:
                parse_shift_name(new_name, "layer_normalization", 25)

            new_var = tf.Variable(pdcd[old_name], name=new_name)
            print(old_name, new_name)
            if new_name in name_set:
                print(new_name)
            assert new_name not in name_set
            name_set.add(new_name)
            var_list.append(new_var)

        sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        saver.save(sess, save_path)


if __name__ == "__main__":
    combine_pdcd_pc_bert(sys.argv[1], sys.argv[2], sys.argv[3])