import os
import pickle
import sys

from arg.perspectives.pc_eval import load_label_from_tfrecord


def summarize(tfrecord_dir, st, ed):
    d = {}
    for i in range(st, ed):
        tf_path = os.path.join(tfrecord_dir, str(i))
        d.update(load_label_from_tfrecord(tf_path))

    pickle.dump(d, open("pc_para_label_{}_{}".format(st, ed), "wb"))


if __name__ =="__main__":
    summarize(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
