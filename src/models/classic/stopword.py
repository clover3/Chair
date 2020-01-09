import os

from cpath import data_path


def load_stopwords():
    s = set()
    #f = open(os.path.join(data_path, "stopwords.dat"), "r")
    f = open(os.path.join(data_path, "smart_stopword.txt"), "r")
    for line in f:
        s.add(line.strip())
    return s