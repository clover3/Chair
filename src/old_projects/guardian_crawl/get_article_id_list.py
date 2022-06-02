import os
import pickle

from crawl.load_guardian import load_article_only_short_url

from cpath import data_path
from misc_lib import get_dir_files, get_dir_dir

scope_dir = os.path.join(data_path, "guardian")


def load():
    root = os.path.join(scope_dir, "by_time")
    l_all = []
    for dir_path in get_dir_dir(root):
        print(dir_path)
        for file_path in get_dir_files(dir_path):
            l = load_article_only_short_url(file_path)
            l_all.extend(l)

    print("Total of {} articles ".format(len(l_all)))
    out_path = os.path.join(root, "list.pickle")
    pickle.dump(l_all, open(out_path, "wb"))





load()