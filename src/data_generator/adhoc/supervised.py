import random
import sys
from misc_lib import *
import pickle

def load_qrel():
    NotImplemented



class DataWriter:
    def __init__(self, max_sequence):
        tprint("Loading data sampler")
        #mem_path = "/dev/shm/robust04.pickle"
        #self.data_sampler = pickle.load(open(mem_path, "rb"))
        self.data_sampler = DataSampler.init_from_pickle("robust04")
        vocab_filename = "bert_voca.txt"
        voca_path = os.path.join(data_path, vocab_filename)
        self.encoder_unit = EncoderUnit(max_sequence, voca_path)
        self.pair_generator = self.data_sampler.pair_generator()


    def encode_pair(self, instance):
        query, case1, case2 = instance
        for y, sent in [case1, case2]:
            entry =  self.encoder_unit.encode_pair(query, sent)
            yield entry["input_ids"], entry["input_mask"], entry["segment_ids"]

    def get_data(self, data_size):
        assert data_size % 2 == 0
        result = []
        ticker = TimeEstimator(data_size, sample_size=100)
        while len(result) < data_size:
            raw_inst = self.pair_generator.__next__()
            result += list(self.encode_pair(raw_inst))
            ticker.tick()
        return result

    def write(self, path, num_data):
        assert num_data % 2 == 0
        pickle.dump(self.get_data(num_data), open(path, "wb"))
