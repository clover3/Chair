from data_generator.common import *
from data_generator.group_sampler import KeySampler, pos_sampling
from data_generator.text_encoder import PAD_ID
from data_generator.text_encoder import SubwordTextEncoder
from misc_lib import slice_n_pad


class AuxPairLoader:
    def __init__(self, seq_length, shared_setting, grouped_data):
        voca_path = os.path.join(data_path, shared_setting.vocab_filename)
        self.voca_size = shared_setting.vocab_size
        self.encoder = SubwordTextEncoder(voca_path)
        self.seq_length = seq_length
        self.grouped_data = grouped_data
        self.sampler = KeySampler(self.grouped_data)

    def encode(self, sent):
        tokens = self.encoder.encode(sent)
        pad_len = self.seq_length - len(tokens)
        return tokens + pad_len * [PAD_ID]

    def case_encoder(self, pair):
        # sent1 : list[int]
        # label : int
        sent1, sent2 = pair
        sent1_enc = slice_n_pad(self.encode(sent1), self.seq_length, PAD_ID)
        sent2_enc = slice_n_pad(self.encode(sent2), self.seq_length, PAD_ID)
        return [(sent1_enc, sent2_enc)]

    def get_insts(self, data_size):
        sent_pairs = pos_sampling(self.grouped_data, self.sampler, data_size)
        generator = [self.case_encoder(p) for p in sent_pairs]
        return list(generator)
