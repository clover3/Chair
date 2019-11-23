import os
from path import model_path
from misc_lib import *
import tensorflow as tf
import logging
import re

tf_logger = logging.getLogger('tensorflow')



def save_model(sess, name, global_step):
    run_dir = os.path.join(model_path, 'runs')
    save_dir = os.path.join(run_dir, name)

    exist_or_mkdir(model_path)
    exist_or_mkdir(run_dir)
    exist_or_mkdir(save_dir)

    path = os.path.join(save_dir, "model")
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1)
    ret = saver.save(sess, path, global_step=global_step)
    tf_logger.info("Model saved at {} - {}".format(path, ret))
    return ret

def load_model(sess, model_path):
    loader = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1)
    loader.restore(sess, model_path)



def load_bert_v2(sess, model_path):
    tvars = tf.contrib.slim.get_variables_to_restore()
    name_to_variable = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(model_path)

    load_mapping = dict()
    for v in init_vars:
        name_tokens = v[0].split('/')
        checkpoint_name = '/'.join(name_tokens).split(":")[0]
        tvar_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", checkpoint_name)
        tvar_name = re.sub("dense[_]?\d*", "dense", tvar_name)
        if tvar_name in name_to_variable:
            tf_logger.debug("{} -> {}".format(checkpoint_name, tvar_name))
            load_mapping[checkpoint_name] = name_to_variable[tvar_name]

    print("Restoring: {}".format(model_path))
    loader = tf.train.Saver(load_mapping, max_to_keep=1)
    loader.restore(sess, model_path)


def load_model_w_scope(sess, path, include_namespace, verbose=True):
    def condition(v):
        if v.name.split('/')[0] in include_namespace:
            return True
        return False

    variables = tf.contrib.slim.get_variables_to_restore()
    variables_to_restore = [v for v in variables if condition(v)]
    if verbose:
        for v in variables_to_restore:
            print(v)

    loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
    loader.restore(sess, path)

