import json
import path
import os
corpus_path = os.path.join(path.data_path, "protest")

def load_data(split):
    json_path = os.path.join(corpus_path, "{}_filled.json".format(split))
    f = open(json_path, "r", encoding="utf-8")

    X = []
    Y = dict()
    for line in f:
        entry = json.loads(line)
        X.append((entry['id'], entry['text']))
        Y[entry['id']] = entry['label']
    return X, Y
