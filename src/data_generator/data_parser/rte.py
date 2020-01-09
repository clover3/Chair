import csv
from collections import Counter

import cpath
from cache import *

corpus_dir = os.path.join(cpath.data_path, "RTE")

def load(split):
    filename = "{}.tsv".format(split)
    path = os.path.join(corpus_dir, filename)
    f = open(path, "r")
    reader = csv.reader(f, delimiter='\t', quotechar=None)

    data = []
    for g_idx, row in enumerate(reader):
        if g_idx ==0 :
            columns = row
        else:
            entry = {}
            for idx, column in enumerate(columns):
                entry[column] = row[idx]
            data.append(entry)

    return data


def tf_stat(data):
    def tokenize(sent):
        if sent[-1] == '.':
            sent = sent[:-1]
        tokens = sent.split()
        return list([t.lower() for t in tokens])

    tf_count = Counter()
    for entry in data:
        for column in ['sentence1', 'sentence2']:
            tokens = tokenize(entry[column])
            tf_count.update(tokens)

    save_to_pickle(tf_count, "rte_tf")


if __name__ == '__main__':
    data = load("train")
    tf_stat(data)