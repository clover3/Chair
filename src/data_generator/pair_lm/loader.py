import csv
import os
import random
import tensorflow as tf
import time


from data_generator.common import *
from data_generator.text_encoder import SubwordTextEncoder
from data_generator.text_encoder import C_MASK_ID, PAD_ID, SEP_ID
from misc_lib import slice_n_pad, TimeEstimator


# grouped_dict : dict(key->list)
def pos_neg_pair_sampling(grouped_dict, key_sampler, target_size):
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
    pos_size = int(target_size / 2)
    neg_size = target_size - pos_size

    data = []
    count = 0
    while count < pos_size:
        key = key_sampler.sample2()
        items = grouped_dict[key]
        i1, i2 = random.sample(range(len(items)), 2)
        data.append((items[i1], items[i2], LABEL_POS))
        count += 1

    for i in range(neg_size):
        key = key_sampler.sample()
        items = grouped_dict[key]
        item1 = items[random.randint(0, len(items)-1)]

        item_2_group = sample_group(key)
        l2 = grouped_dict[item_2_group]
        item_2_idx = random.randint(0, len(l2)-1)
        item2 = l2[item_2_idx]
        data.append((item1, item2, LABEL_NEG))

    assert len(data) == target_size
    random.shuffle(data)
    return data


class PairDataLoader():
    def __init__(self, seq_length, shared_setting, grouped_data):
        voca_path = os.path.join(data_path, shared_setting.vocab_filename)
        self.voca_size = shared_setting.vocab_size
        self.encoder = SubwordTextEncoder(voca_path)
        self.seq_length = seq_length
        self.mask_rate = 0.15
        self.grouped_data = grouped_data
        self.train_group = None
        self.test_group = None

    def encode(self, sent):
        tokens = self.encoder.encode(sent)
        pad_len = self.seq_length - len(tokens)
        return tokens + pad_len * [PAD_ID]

    def delete(self, sent):
        n_delete = int(self.seq_length * self.mask_rate)
        delete_indice = random.sample(range(self.seq_length), n_delete)
        x = list(sent)
        y = list(sent)
        for idx in delete_indice:
            action = random.randrange(0, 10)
            if action < 8:
                x[idx] = C_MASK_ID
            elif action == 8:
                rand_char = random.randrange(0, self.voca_size)
                x[idx] = rand_char
            else:
                pass
        return x, y


    def case_encoder(self, plain_insts):
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

    @staticmethod
    def split_dict(d, held_out_size):
        keys = list(d.keys())
        indice = random.sample(range(0, len(keys)), held_out_size)
        held_out_keys = [keys[i] for i in indice]

        train_d = {}
        test_d = {}
        for key, items in d.items():
            if key in held_out_keys:
                test_d[key] = items
            else:
                train_d[key] = items
        return train_d, test_d

    def split_train_test(self):
        print("split_train_test 1")
        held_out_group = 4000
        self.train_group, self.test_group = self.split_dict(self.grouped_data, held_out_group)
        print("split_train_test 2")


        class KeySampler:
            def __init__(self, group_dict):
                self.interval = [1,4,16,64] # last is assumped to be 64

                def find_group(n):
                    last_group = len(self.interval) - 1
                    for i in range(last_group):
                        if n <= self.interval[i] :
                            return i
                        return last_group

                self.sample_groups = list([list() for i in self.interval])
                for key, items in group_dict.items():
                    gid = find_group(len(items))
                    self.sample_groups[gid].append(key)

                self.sample_prior = []
                for gid, g, in enumerate(self.sample_groups):
                    a = self.interval[gid]
                    sample_size = a * a * len(g)
                    self.sample_prior.append(sample_size)
                s = sum(self.sample_prior)
                for i, v in enumerate(self.sample_prior):
                    self.sample_prior[i] = v / s

                s2 = sum(self.sample_prior[1:])
                self.sample_prior2 = []
                for i, v in enumerate(self.sample_prior[1:]):
                    self.sample_prior2.append(v / s2)

            # Sample from all group
            def sample(self):
                def select_group():
                    dice = random.random()
                    acc =0
                    for i, v in enumerate(self.sample_prior):
                        acc += v
                        if dice < acc:
                            return i
                    return len(self.sample_prior)-1

                sample_space = self.sample_groups[select_group()]
                end = len(sample_space) - 1
                return sample_space[random.randint(0, end)]

            # Sample from groups except first size
            def sample2(self):
                def select_group():
                    dice = random.random()
                    acc =0
                    for i, v in enumerate(self.sample_prior2):
                        acc += v
                        if dice < acc:
                            return i
                    return len(self.sample_prior2)-1

                sample_space = self.sample_groups[1+select_group()]
                end = len(sample_space) - 1
                return sample_space[random.randint(0, end)]

        self.test_sampler = KeySampler(self.test_group)
        print("split_train_test 3")
        self.train_sampler = KeySampler(self.train_group)
        print("split_train_test 4")


    # Child classs will feed own text to case_generator
    # and return generator of x,y tuples
    def get_train_batch(self, data_size):
        if self.train_group is None:
            self.split_train_test()
        train_generator = self.case_encoder(pos_neg_pair_sampling(self.train_group,
                                                                  self.train_sampler,
                                                                  data_size))
        return train_generator

    def get_test_generator(self, data_size):
        if self.test_group is None:
            self.split_train_test()
        test_generator = self.case_encoder(pos_neg_pair_sampling(self.test_group,
                                                                 self.test_sampler,
                                                                 data_size))
        return test_generator
