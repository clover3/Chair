import pickle
import path
import os
from misc_lib import pair_shuffle
from trainer.np_modules import get_batches_ex
import random

class DataLoader:
    def __init__(self, max_seq, dim):
        self.max_seq = max_seq
        self.dim = dim
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        item_path = os.path.join(path.data_path, "cache", "items.pickle")
        self.items = pickle.load(open(item_path, "rb"))
        random.shuffle(self.items)
        self.idx = 0

    def get_train_batch(self, batch_size):
        result = self.get_data(batch_size)
        batches = get_batches_ex(result, batch_size, 3)
        return batches[0]

    def get_data(self, data_size):
        assert data_size % 2 == 0
        result = []
        while len(result) < data_size:
            raw_inst = self.items[self.idx]
            self.idx += 1
            result += list(self.encode_pair(raw_inst))
        return result

    def get_dev_data(self, batch_size):
        result = self.get_data(batch_size * 10)
        batches = get_batches_ex(result, batch_size, 3)
        return batches

    def encode_pair(self, raw_inst):
        for v in raw_inst:
            yield encode(v, self.max_seq, self.dim)




def encode(vector_list, max_seq, dim):
    vector_list = vector_list[:max_seq-2]
    zero_vector = dim * [0]

    tokens = []
    segment_ids = []
    tokens.append(zero_vector)
    segment_ids.append(0)
    for vector in vector_list:
        assert len(vector) < dim
        padded_vector = vector + (dim - len(vector)) * [0]
        tokens.append(padded_vector)
        segment_ids.append(1)
    tokens.append(zero_vector)
    segment_ids.append(0)

    input_vectors = tokens
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_vectors)


    # Zero-pad up to the sequence length.
    while len(input_vectors) < max_seq:
        input_vectors.append(zero_vector)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_vectors) == max_seq
    assert len(input_mask) == max_seq
    assert len(segment_ids) == max_seq

    #return {
    #    "input_ids": input_vectors,
    #    "input_mask": input_mask,
    #    "segment_ids": segment_ids
    #}
    return input_vectors, input_mask, segment_ids
