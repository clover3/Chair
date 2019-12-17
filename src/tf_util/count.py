import os
import sys

import tensorflow as tf


def file_cnt(fn):
    c = 0
    for record in tf.python_io.tf_record_iterator(fn):
       c += 1
    return c


def count_instance(dir_name):
    acc = 0
    for _,_,filenames in os.walk(dir_name):
        for fn in filenames:
            path = os.path.join(dir_name, fn)
            cnt = file_cnt(path)
            print(fn, cnt)
            acc += cnt
    return acc


if __name__ == "__main__":
    print("Number of record : ", file_cnt(sys.argv[1]))
