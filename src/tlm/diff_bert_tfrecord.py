import os

import numpy as np

from cache import load_cache
from cpath import data_path, output_path
from tf_util.count import file_cnt
from tf_util.enum_features import load_record


def get_iterator():
    return load_record(os.path.join(data_path, "nli", "bert_code_train.tf_record"))


def count_compare():
    a = file_cnt(os.path.join(data_path, "nli", "bert_code_train.tf_record"))
    b = file_cnt(os.path.join(output_path, "nli_tfrecord_cls_300", "train"))
    print(a, b)

def compare():
    bert_itr = get_iterator()
    my_itr = load_cache("nli_train_cache")
    #my_itr = load_record(os.path.join(output_path, "nli_tfrecord_cls_300", "train"))

    cnt = 0

    for f1, f2 in zip(bert_itr, my_itr):
        b1 = f1['input_ids'].int64_list.value
        input_ids, input_mask, segment_ids, l = f2
        if not np.all(np.equal(b1, input_ids)):
            print(cnt)
            for i in range(len(b1)):
                if b1[i] != input_ids[i] :
                    print(b1[i], input_ids[i])
            print(b1)
            print(input_ids)

        cnt += 1

if __name__ =="__main__":
    compare()