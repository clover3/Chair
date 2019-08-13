import tensorflow as tf

def file_show(fn):
    for record in tf.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        print(example)