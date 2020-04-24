import os
import re

import tensorflow as tf

from cpath import output_path
from misc.show_checkpoint_vars import load_checkpoint_vars


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def rewrite(checkpoint, save_path):
    # Write variables names of nli_checkpoint on alt_emb_checkpoint
    checkpoint_vars = load_checkpoint_vars(checkpoint)
    var_list = []
    layer_norm_idx = 0
    layer_prefix_d = {}
    dense_prefix_d = {}
    dense_idx = 0
    with tf.Session() as sess:
        keys = list(checkpoint_vars.keys())
        keys.sort(key=natural_keys)
        for key in keys:
            print(key)
            if "LayerNorm" in key:
                prefix = key[:key.find("LayerNorm")]
                if prefix in layer_prefix_d:
                    layer_norm = layer_prefix_d[prefix]
                else:
                    layer_norm = "layer_normalization"
                    if layer_norm_idx > 0:
                        layer_norm += "_{}".format(layer_norm_idx)
                    layer_prefix_d[prefix] = layer_norm
                    layer_norm_idx += 1
                new_name = re.sub("LayerNorm", layer_norm, key)
            elif "dense" in key:
                prefix = key[:key.find("dense")]
                if prefix in dense_prefix_d:
                    dense = dense_prefix_d[prefix]
                else:
                    dense = "dense"
                    if dense_idx > 0:
                        dense += "_{}".format(dense_idx)
                    dense_idx += 1
                    dense_prefix_d[prefix] = dense
                new_name = re.sub("dense", dense, key)
            else:
                new_name = key


            new_var = tf.Variable(checkpoint_vars[key], name=new_name)
            var_list.append(new_var)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, save_path)


if __name__ == "__main__":
    save_path = os.path.join(output_path, "model", "msmarco_2", "msmarco_2")
    rewrite("C:\work\Code\Chair\output\model\BERT_Base_trained_on_MSMARCO\\model.ckpt-100000",
            save_path)