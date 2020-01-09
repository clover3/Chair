import os
import pickle
import random
import threading
from multiprocessing import Queue

import cpath
from config.input_path import train_data_dir
from trainer.np_modules import get_batches_ex


class DataLoader:
    def __init__(self, max_seq, dim):
        self.max_seq = max_seq
        self.dim = dim
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        item_path = os.path.join(cpath.data_path, "cache", "items.pickle")
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
        if len(vector) < dim:
            padded_vector = list(vector) + (dim - len(vector)) * [0]
        else:
            padded_vector = vector[:dim]
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


class NetOutputLoader:
    # Each pickle List[Pair[output1, output2]]
    # output = List[score, vector]
    def __init__(self, max_seq, dim, batch_size):
        self.max_seq = max_seq
        self.dim = dim
        self.batch_size = batch_size
        self.train_queue = Queue(maxsize=20)
        t = threading.Thread(target=self.feed_queue)
        t.daemon = True
        t.start()

        self.cur_idx = 0
        self.file_idx = 0
        self.cur_data = []
        self.load_next_data()

    def get_path(self, i):
        filename = "merger_train_{}.pickle.output".format(i)
        return os.path.join(train_data_dir, filename)

    def feed_queue(self):
        print("feed_queue()")
        while True:
            batches = self.get_data(self.batch_size, 10)
            for batch in batches:
                self.train_queue.put(batch, True)

    def get_dev_data(self, batch_size):
        n_batches = 10
        return self.get_data(batch_size, n_batches)

    def load_next_data(self):
        path = self.get_path(self.file_idx)
        self.file_idx += 1
        next_path = self.get_path(self.file_idx)
        if not os.path.exists(next_path):
            print("WARNING next file is unavailable : {}".format(next_path))
        self.cur_data = pickle.load(open(path, "rb"))
        print("Loaded data {}".format(self.file_idx - 1))
        self.cur_idx = 0
        return self.cur_data

    def encode(self, t):
        output1, output2 = t
        result = []
        for output in [output1, output2]:
            out_v_1 = []
            for score, vector in output:
                #new_v = [score]
                new_v = [score] + vector
                out_v_1.append(new_v)
            out_v_1.sort(key=lambda x:x[0], reverse=True)
            result.append(encode(out_v_1, self.max_seq, self.dim))

        return result

    def get_data(self, batch_size, n_batches):
        st = self.cur_idx
        ed = self.cur_idx + int((batch_size * n_batches) / 2)
        if ed > len(self.cur_data):
            self.cur_data = self.load_next_data()
            st = self.cur_idx
            ed = self.cur_idx + batch_size * n_batches
        raw_data = self.cur_data[st:ed]

        enc_data = []
        for t in raw_data:
            enc_data += self.encode(t)
        self.cur_idx = ed
        batches = get_batches_ex(enc_data, batch_size, 3)
        return batches

    def get_train_batch(self, batch_size):
        return self.train_queue.get(block=True)
