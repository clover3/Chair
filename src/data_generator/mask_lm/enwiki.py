import os
from data_generator.common import *
corpus_dir = os.path.join(data_path, "enwiki")
train_path = os.path.join(corpus_dir, "enwiki_train.txt")
eval_path = os.path.join(corpus_dir, "enwiki_eval.txt")

from data_generator.mask_lm.chunk_lm import DataLoader


class EnwikiLoader(DataLoader):
    def __init__(self, seq_length, shared_setting):
        super(EnwikiLoader).__init__(seq_length, shared_setting)

    def get_train_generator(self):
        reader = tf.gfile.Open(train_path, "r")
        return self.case_generator(reader)

    def get_test_generator(self):
        reader = tf.gfile.Open(eval_path, "r")
        return self.case_generator(reader)
