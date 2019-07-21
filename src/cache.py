from path import cache_path, data_path
import os
import pickle


def load_from_pickle(name):
    pickle_name = "{}.pickle".format(name)
    path = os.path.join(cache_path, pickle_name)
    return pickle.load(open(path, "rb"))


def save_to_pickle(obj, name):
    assert type(name) == str
    pickle_name = "{}.pickle".format(name)
    path = os.path.join(cache_path, pickle_name)
    pickle.dump(obj, open(path, "wb"))


def load_cache(name):
    pickle_name = "{}.pickle".format(name)
    path = os.path.join(cache_path, pickle_name)
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    else:
        return None


class StreamPickler:
    def __init__(self, name, save_per):
        self.idx = 0
        self.current_chunk = []
        self.save_per = save_per
        self.save_prefix = os.path.join(data_path, "stream_pickled", name)

    def flush(self):
        if len(self.current_chunk) == 0:
            return
        save_name = self.save_prefix + str(self.idx)
        pickle.dump(self.current_chunk, open(save_name, "wb"))
        self.current_chunk = []
        self.idx += 1

    def add(self, inst):
        self.current_chunk.append(inst)
        if len(self.current_chunk) == self.save_per :
            self.flush()


class StreamPickleReader:
    def __init__(self, name, pickle_idx = 0):
        self.pickle_idx = pickle_idx
        self.current_chunk = []
        self.chunk_idx = 0
        self.save_prefix = os.path.join(data_path, "stream_pickled", name)

    def get_item(self):
        if self.chunk_idx >= len(self.current_chunk):
            self.get_new_chunk()

        item = self.current_chunk[self.chunk_idx]
        self.chunk_idx += 1
        return item

    def get_new_chunk(self):
        save_name = self.next_chunk_path()
        self.current_chunk = pickle.load(open(save_name, "rb"))
        assert len(self.current_chunk) > 0
        self.chunk_idx = 0
        self.pickle_idx += 1

    def next_chunk_path(self):
        return self.save_prefix + str(self.pickle_idx)

    def has_next(self):
        if self.chunk_idx +1 < len(self.current_chunk):
            return True

        return os.path.exists(self.next_chunk_path())
