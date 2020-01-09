import random

from cpath import cache_path
from data_generator.common import *
from data_generator.group_sampler import KeySampler
from data_generator.text_encoder import C_MASK_ID, PAD_ID, SEP_ID
from data_generator.text_encoder import SubwordTextEncoder
from misc_lib import slice_n_pad, increment_circular


class AuthorAsDoc:
    def __init__(self, seq_length, shared_setting, grouped_data):
        voca_path = os.path.join(data_path, shared_setting.vocab_filename)
        self.voca_size = shared_setting.vocab_size
        self.encoder = SubwordTextEncoder(voca_path)
        self.seq_length = seq_length
        self.grouped_data = grouped_data
        self.train_group = None
        self.test_group = None
        self.test_sampler = None
        self.train_sampler = None
        self.mask_rate = 0.15

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

    def index_data(self):
        if self.test_group is None:
            self.split_train_test()


    def split_train_test(self):
        print("split_train_test 1")
        held_out_group = 4000
        self.train_group, self.test_group = self.split_dict(self.grouped_data, held_out_group)
        print("split_train_test 2")

        self.test_sampler = KeySampler(self.test_group)
        print("split_train_test 3")
        self.train_sampler = KeySampler(self.train_group)
        print("split_train_test 4")



    @classmethod
    def load_from_pickle(cls, id):
        pickle_name = "AuthorAsDoc_{}".format(id)
        path = os.path.join(cache_path, pickle_name)
        return pickle.load(open(path, "rb"))

    def save_to_pickle(self, id):
        pickle_name = "AuthorAsDoc_{}".format(id)
        path = os.path.join(cache_path, pickle_name)
        pickle.dump(self, open(path, "wb"))

    def encode(self, sent):
        tokens = self.encoder.encode(sent)
        return tokens + [SEP_ID]

    def delete_alter(self, sent):
        n_delete = int(self.seq_length * self.mask_rate)
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
        return x, y

    def get_instances(self, grouped_dict, key_sampler, data_size):
        data = []
        for i in range(data_size):
            key = key_sampler.sample2()
            items = grouped_dict[key]
            seq = []
            j_init = random.randint(0, len(items)-1)
            j = 0
            while len(seq) < self.seq_length:
                sent = self.encode(items[j])
                if len(seq) + len(sent) > self.seq_length:
                    break
                seq += sent
                j = increment_circular(j, len(items))
                if j == j_init:
                    break

            seq = slice_n_pad(seq, self.seq_length, PAD_ID)
            data.append(self.delete_alter(seq))
        return data

    def get_train_instances(self, data_size):
        return self.get_instances(self.train_group, self.train_sampler, data_size)

    def get_test_instances(self, data_size):
        return self.get_instances(self.test_group, self.test_sampler, data_size)
