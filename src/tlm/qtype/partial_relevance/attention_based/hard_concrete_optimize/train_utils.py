import os

import tensorflow as tf

from models.bert_util.bert_utils import get_last_id


def session_run_print(sess, feed_dict, fetch_tensors_d):
    fetched_values_d = session_run(sess, feed_dict, fetch_tensors_d)
    print_dict_float_values(fetched_values_d)
    return fetched_values_d


def session_run(sess, feed_dict, fetch_tensors_d):
    fetch_tensors_keys = list(fetch_tensors_d.keys())
    fetch_tensors_list = [fetch_tensors_d[k] for k in fetch_tensors_keys]
    fetched_values = sess.run(fetch_tensors_list, feed_dict=feed_dict)
    fetched_values_d = dict(zip(fetch_tensors_keys, fetched_values))
    return fetched_values_d


def print_dict_float_values(fetched_values_d):
    s_list = []
    for k, v in fetched_values_d.items():
        try:
            s_list.append("{0}: {1:.2f}".format(k, v))
        except TypeError:
            pass
    print(", ".join(s_list))


def load_model_by_dir(sess, save_dir, variables_to_restore=None):
    id = get_last_id(save_dir)
    path = os.path.join(save_dir, "{}".format(id))
    if variables_to_restore is not None:
        loader = tf.compat.v1.train.Saver(variables_to_restore, max_to_keep=1)
    else:
        loader = tf.compat.v1.train.Saver(max_to_keep=1)
    loader.restore(sess, path)
