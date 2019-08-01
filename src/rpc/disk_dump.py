import path
import os
import pickle


dump_root = '/mnt/scratch/youngwookim/data'
if not os.path.exists(dump_root):
    dump_root = '/mnt/nfs/work3/youngwookim/data'

def dump_dict(d, name):
    dir_path = os.path.join(dump_root, "name_dump", name)
    os.mkdir(dir_path)
    for key in d:
        p = os.path.join(dir_path, key)
        pickle.dump(d[key], open(p,"wb"))



class DumpAccess:
    def __init__(self, name):
        self.dir_path = os.path.join(dump_root, "name_dump", name)

    def get(self, key):
        p = os.path.join(self.dir_path, key)
        return pickle.load(open(p, "rb"))

