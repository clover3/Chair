import csv
import os
import random
import tensorflow as tf

from data_generator.common import *
from data_generator.text_encoder import SubwordTextEncoder
from data_generator.text_encoder import C_MASK_ID, PAD_ID, SEP_ID
from misc_lib import slice_n_pad


# grouped_dict : dict(key->list)
def pos_neg_pair_sampling(grouped_dict, target_size = 0):
    pos_pair_size = {k: len(v)*len(v) for k,v in grouped_dict.items()}
    total_complexity = sum(pos_pair_size.values())

    if target_size == 0:
        target_size = total_complexity / 10

    # 0.1 for default setting
    sample_per_complexity = target_size / total_complexity

    # from default setting, only group with more than 3 items will be sampled

    group_names = list(grouped_dict.keys())

    def sample_group(except_key = None):
        gid = random.randint(0, len(group_names)-1)
        sample_name = group_names[gid]
        while except_key is not None and except_key == sample_name:
            gid = random.randint(0, len(group_names) - 1)
            sample_name = group_names[gid]

        return sample_name

    LABEL_POS = 1
    LABEL_NEG = 0
    for key in pos_pair_size.keys():
        items = grouped_dict[key]
        complexity = len(items) * len(items)
        sample_size = int(complexity * sample_per_complexity)
        pos_size = int(sample_size / 2)
        for i in range(pos_size):
            i1, i2 = random.sample(range(len(items)), 2)
            yield items[i1], items[i2], LABEL_POS

        neg_size = int(sample_size / 2)
        for i in range(neg_size):
            item1 = items[random.randint(0, len(items)-1)]

            item_2_group = sample_group(key)
            l2 = grouped_dict[item_2_group]
            item_2_idx = random.randint(0, len(l2)-1)
            item2 = l2[item_2_idx]
            yield item1, item2, LABEL_NEG


class DataLoader():
    def __init__(self, seq_length, shared_setting, grouped_data):
        voca_path = os.path.join(data_path, shared_setting.vocab_filename)
        self.voca_size = shared_setting.vocab_size
        self.encoder = SubwordTextEncoder(voca_path)
        self.seq_length = seq_length
        self.mask_rate = 0.15
        self.grouped_data = grouped_data
        self.train_data = None
        self.test_data = None

    def encode(self, sent):
        tokens = self.encoder.encode(sent)
        pad_len = self.seq_length - len(tokens)
        return tokens + pad_len * [PAD_ID]

    def delete(self, sent):
        n_delete = int(self.seq_length * self.mask_rate)
        delete_indice = random.sample(range(self.seq_length), n_delete)
        x = list(sent)
        y = [0 for i in sent]
        for idx in delete_indice:
            action = random.randrange(0, 10)
            if action < 8:
                x[idx] = C_MASK_ID
            elif action == 8:
                rand_char = random.randrange(0, self.voca_size)
                x[idx] = rand_char
            else:
                pass
            y[idx] = sent[idx]
        return x, y


    def case_generator(self, plain_insts):
        random.seed(0)
        # sent1 : list[int]
        # label : int

        for sent1, sent2, label in plain_insts:
            sent1_enc = slice_n_pad(self.encode(sent1), self.seq_length, PAD_ID)
            sent2_enc = slice_n_pad(self.encode(sent2), self.seq_length, PAD_ID)

            sent1_del, y_1 = self.delete(sent1_enc)
            sent2_del, y_2 = self.delete(sent2_enc)
            x = sent1_del + [SEP_ID] + sent2_del
            y_seq = y_1 + [0] + y_2
            y_cls = label
            yield x, y_seq, y_cls

    def sample_and_encode(self):
        generator = self.case_generator(pos_neg_pair_sampling(self.grouped_data))
        self.all_inst = list(generator)

        random.shuffle(self.all_inst)

        train_size = int(0.9 * len(self.all_inst))
        self.train_data = self.all_inst[:train_size]
        self.test_data = self.all_inst[train_size:]

    # Child classs will feed own text to case_generator
    # and return generator of x,y tuples
    def get_train_generator(self):
        if self.train_data is None:
            self.sample_and_encode()
        return self.train_data

    def get_test_generator(self):
        if self.test_data is None:
            self.sample_and_encode()
        return self.test_data
