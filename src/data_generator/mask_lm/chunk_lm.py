
import csv
import os
import tensorflow as tf
from data_generator.common import *
from data_generator.text_encoder import SubwordTextEncoder
from data_generator.text_encoder import C_MASK_ID


import random

class DataLoader():
    def __init__(self, seq_length, shared_setting):
        voca_path = os.path.join(data_path, shared_setting.vocab_filename)
        self.voca_size = shared_setting.vocab_size
        self.encoder = SubwordTextEncoder(voca_path)
        self.seq_length = seq_length
        self.mask_rate = 0.15

    def token_generator(self, reader):
        buf = []
        for line in reader:
            tokens = self.encoder.encode(line)

            buf.extend(tokens)

            if len(buf) > self.seq_length:
                yield buf[:self.seq_length]
                buf = buf[self.seq_length:]

    def case_generator(self, reader):
        sents = self.token_generator(reader)
        random.seed(0)

        n_delete = int(self.seq_length * self.mask_rate)
        for sent in sents:
            delete_indice = random.sample(range(self.seq_length), n_delete)
            x = list(sent)
            for idx in delete_indice:
                action = random.randrange(0,10)
                if action < 8:
                    x[idx] = C_MASK_ID
                elif action == 8 :
                    rand_char = random.randrange(0, self.voca_size)
                    x[idx] = rand_char
                else:
                    pass
            y = list(sent)
            yield x, y

    # Child classs will feed own text to case_generator
    # and return generator of x,y tuples
    def get_train_generator(self):
        raise NotImplementedError()

    def get_test_generator(self):
        raise NotImplementedError()


class Text2Case(DataLoader):
    def __init__(self, text_list, seq_length, shared_setting):
        text_list = list(text_list)
        super(Text2Case, self).__init__(seq_length, shared_setting)
        train_size = int(0.9 * len(text_list))
        self.train_data = text_list[:train_size]
        self.test_data = text_list[train_size:]

    def get_train_generator(self):
        return self.case_generator(self.train_data)

    def get_test_generator(self):
        return self.case_generator(self.test_data)


