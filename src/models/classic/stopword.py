import os

from cpath import data_path


def load_stopwords():
    s = set()
    #f = open(os.path.join(data_path, "stopwords.dat"), "r")
    f = open(os.path.join(data_path, "smart_stopword.txt"), "r")
    for line in f:
        s.add(line.strip())
    return s


def load_stopwords_ex():
    s = load_stopwords()
    s.update(get_all_punct())
    return s


def get_all_punct():
    punct_range = [[33, 48], [58, 65], [91, 97], [123, 127]]
    r = []
    for st, ed in punct_range:
        for j in range(st, ed):
            c = chr(j)
            r.append(c)

    return r


loaded_stopword = None


def is_stopword(word):
    global loaded_stopword
    if loaded_stopword is None:
        loaded_stopword = load_stopwords()

    return word in loaded_stopword

