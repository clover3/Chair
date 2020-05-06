import os

from arg.perspectives.baseline_bert_gen import baseline_bert_gen
from cpath import data_path
from misc_lib import exist_or_mkdir


def make_train():
    dir_path = os.path.join(data_path, "perspective_bert_tfrecord")
    exist_or_mkdir(dir_path)
    baseline_bert_gen(os.path.join(dir_path, "train"), "train")
    baseline_bert_gen(os.path.join(dir_path, "dev"), "dev")


if __name__ == "__main__":
    make_train()
