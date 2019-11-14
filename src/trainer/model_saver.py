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
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    ret = saver.save(sess, path, global_step=global_step)
    tf_logger.info("Model saved at {} - {}".format(path, ret))
    return ret


def load_bert_v2(sess, dir_name, file_name):
    run_dir = os.path.join(model_path, 'runs')
    save_dir = os.path.join(run_dir, dir_name)
    path = os.path.join(save_dir, file_name)

    tvars = tf.contrib.slim.get_variables_to_restore()
    name_to_variable = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(path)

    load_mapping = dict()
    for v in init_vars:
        name_tokens = v[0].split('/')
        checkpoint_name = '/'.join(name_tokens).split(":")[0]
        tvar_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", checkpoint_name)
        tvar_name = re.sub("dense[_]?\d*", "dense", tvar_name)
        if tvar_name in name_to_variable:
            tf_logger.debug("{} -> {}".format(checkpoint_name, tvar_name))
            load_mapping[checkpoint_name] = name_to_variable[tvar_name]

    print("Restoring: {} {}".format(dir_name, file_name))
    loader = tf.train.Saver(load_mapping, max_to_keep=1)
    loader.restore(sess, path)
