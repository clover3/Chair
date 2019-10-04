import tensorflow as tf


def read_bert_data(fn):
    for record in tf.python_io.tf_record_iterator(fn):
        print(record.features)



def dev():
    path = "C:\work\code\chair\data\\0"
    read_bert_data(path)

if __name__ == "__main__":

    dev()

