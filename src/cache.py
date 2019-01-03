from path import cache_path
import os
import pickle


def load_from_pickle(name):
    pickle_name = "{}.pickle".format(name)
    path = os.path.join(cache_path, pickle_name)
    return pickle.load(open(path, "rb"))


def save_to_pickle(obj, name):
    pickle_name = "{}.pickle".format(name)
    path = os.path.join(cache_path, pickle_name)
    pickle.dump(obj, open(path, "wb"))
