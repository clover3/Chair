import os
import csv
import path

corpus_dir = os.path.join(path.data_path, "ukp")

def load(topic):
    filename = "{}.tsv".format(topic)
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