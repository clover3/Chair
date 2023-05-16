import tensorflow as tf


def get_pos_only_weight_param(shape, name):
    output_weights = tf.compat.v1.get_variable(
        name, shape,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
    )
    return tf.sigmoid(output_weights)


def show_tfrecord(fn, n_display=5):
    cnt = 0
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        keys = feature.keys()

        print("---- record -----")
        for key in keys:
            if key in ["masked_lm_weights", "rel_score"]:
                v = feature[key].float_list.value
            else:
                v = feature[key].int64_list.value
            print(key)
            print(v)

        cnt += 1
        if cnt >= n_display:  ##
            break


def reduce_max(tensor, axis_arr):
    for axis in axis_arr:
        tensor = tf.reduce_max(tensor, axis)
    return tensor


def find_layer(model, name):
    for l in model.layers:
        if l.name == name:
            return l
    raise KeyError("{} is not found".format(name))