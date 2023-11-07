import sys
import tensorflow as tf
import numpy as np

from misc_lib import Averager


def take(v):
    return np.array(v.int64_list.value)


def check(fn):
    mask_id = 103
    itr = tf.data.TFRecordDataset(fn)
    avg = Averager()
    for record in itr:
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        feature = example.features.feature
        input_ids = take(feature["input_ids"])
        segment_ids = take(feature["segment_ids"])
        n_evidence = np.sum(segment_ids)
        n_mask = np.count_nonzero(np.equal(mask_id, input_ids * segment_ids))
        avg.append(n_mask/n_evidence)
    print(avg.get_average())


def main():
    check(sys.argv[1])


if __name__ == "__main__":
    main()