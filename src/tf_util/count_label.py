import os
import sys
from collections import Counter

from tf_v2_support import tf1

tf = tf1


def count_for_data(fn):
    counter = Counter()
    for record in tf.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        label_ids = feature["label_ids"].int64_list.value
        label = label_ids[0]
        counter[label] += 1

    return counter


def count_for_dir(dir_path, idx_range):
    c_all = Counter()
    for i in idx_range:
        print("Reading {}".format(i))
        fn = os.path.join(dir_path, str(i))
        if os.path.exists(fn):
            c = count_for_data(fn)

            for key, cnt in c.items():
                c_all[key] += cnt

    return c_all

if __name__ == "__main__":
    if len(sys.argv) == 4:
        dir_path = sys.argv[1]
        st = int(sys.argv[2])
        ed = int(sys.argv[3])
        c = count_for_dir(dir_path, range(st, ed))
        print(c)
    elif len(sys.argv) == 2:
        p = sys.argv[1]
        c = count_for_data(p)
        print(c)

