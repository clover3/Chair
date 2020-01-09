import os
import sys

import tensorflow as tf


def validate_data(fn):
    keys = None
    length_d ={}
    for record in tf.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        if keys is None:
            keys = list(feature.keys())
        for key in keys:
            if key == "masked_lm_weights":
                v = feature[key].float_list.value
            else:
                v = feature[key].int64_list.value

            if key not in length_d:
                length_d[key] = len(v)
            if length_d[key] == len(v):
                pass
            else:
                print("Error at {}".format(fn))
    print(keys)

def validate_dir(dir_path, idx_range):
    corrupt_list = []
    for i in idx_range:
        print("Check {}".format(i))
        fn = os.path.join(dir_path, str(i))
        if os.path.exists(fn):
            try:
                validate_data(fn)
            except Exception as e:
                corrupt_list.append(i)

        else:
            print("WARNING data {} doesn't exist".format(i))

    if corrupt_list:
        print("Corrputed _list")
    for e in corrupt_list:
        print(e)

if __name__ == "__main__":
    if len(sys.argv) == 4:
        dir_path = os.path.join("/mnt/nfs/work3/youngwookim/data/bert_tf/", sys.argv[1])
        st = int(sys.argv[2])
        ed = int(sys.argv[3])
        validate_dir(dir_path, range(st, ed))
    elif len(sys.argv) == 2:
        p = sys.argv[1]
        validate_data(p)