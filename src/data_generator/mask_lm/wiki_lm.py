
import csv
import os
import tensorflow as tf
from data_generator.common import *
from data_generator.text_encoder import SubwordTextEncoder
from data_generator.text_encoder import C_MASK_ID

corpus_dir = os.path.join(data_path, "enwiki")
train_path = os.path.join(corpus_dir, "enwiki_train.txt")
eval_path = os.path.join(corpus_dir, "enwiki_eval.txt")

from data_generator.shared_setting import *

import random

class DataLoader():
    def __init__(self, seq_length):
        voca_path = os.path.join(data_path, vocab_filename)
        self.encoder = SubwordTextEncoder(voca_path)
        self.seq_length = seq_length
        self.mask_rate = 0.15

    def token_generator(self, corpus_path):
        reader = tf.gfile.Open(corpus_path, "r")
        buf = []
        for line in reader:
            tokens = self.encoder.encode(line)

            buf.extend(tokens)

            if len(buf) > self.seq_length:
                yield buf[:self.seq_length]
                buf = buf[self.seq_length:]

    def case_generator(self, corpus_path):
        sents = self.token_generator(corpus_path)
        random.seed(0)

        n_delete = int(self.seq_length * self.mask_rate)

        for sent in sents:
            delete_indice = random.sample(range(self.seq_length), n_delete)
            x = list(sent)
            y = [0 for i in sent]
            for idx in delete_indice:
                action = random.randrange(0,10)
                if action < 8:
                    x[idx] = C_MASK_ID
                elif action == 8 :
                    rand_char = random.randrange(0, vocab_size)
                    x[idx] = rand_char
                else:
                    pass
                y[idx] = sent[idx]
            yield x, y

    def get_train_generator(self):
        return self.case_generator(train_path)

    def get_test_generator(self):
        return self.case_generator(eval_path)
