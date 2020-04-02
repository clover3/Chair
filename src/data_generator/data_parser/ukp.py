import csv
import os

import cpath
from base_type import *

corpus_dir = os.path.join(cpath.data_path, "ukp")


def load(topic: str) -> List[Dict]:
    filename = "{}.tsv".format(topic)
    path = os.path.join(corpus_dir, filename)
    f = open(path, "r", encoding="utf-8")
    reader = csv.reader(f, delimiter='\t', quotechar=None)

    data = []
    for g_idx, row in enumerate(reader):
        if g_idx ==0 :
            columns = row
        else:
            entry: Dict = {}
            for idx, column in enumerate(columns):
                entry[column] = row[idx]
            data.append(entry)

    return data