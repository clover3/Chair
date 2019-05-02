import xml.sax
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from path import *
from misc_lib import *
import math
import os
from cache import *
import glob
import re
import json

scope_dir = os.path.join(data_path, "controversy")
corpus_dir = os.path.join(scope_dir, "Amsterdam_data")
#corpus_dir = os.path.join(scope_dir, "web_hard")

collection_types = ['web', 'total', 'wiki']
name_format = "{}_{}_{}_final.json"
collection_type = collection_types[1]


def load_json(path):
    obj = json.load(open(path, "r"))
    data = []
    for doc_id in obj:
        title, content, label = obj[doc_id]
        e = {
            "title": title,
            "doc_id":doc_id,
            "content": content,
            "label":label,
        }
        data.append(e)
    return data

def load_data_split(split):
    data = []
    for label in ["pos", "neg"]:
        name = name_format.format(collection_type, split, label)
        path = os.path.join(corpus_dir, split, label, name)
        data.extend(load_json(path))
    return data

def load_data_split_separate(split):
    data = []
    for label in ["pos", "neg"]:
        name = name_format.format(collection_type, split, label)
        path = os.path.join(corpus_dir, split, label, name)
        data.append(load_json(path))
    return data



def get_train_data(separate = False):
    if separate:
        return load_data_split_separate("train")
    else:
        return load_data_split("train")

def get_dev_data(all_data = True):
    if all_data:
        return load_data_split("val")
    else:
        entries = load_data_split("val")
        dev_X = list([entry["title"] + "\t" + entry["content"] for entry in entries])
        dev_Y = list([entry["label"] for entry in entries])
        return dev_X, dev_Y




if __name__ == '__main__':
    print(len(get_train_data()))
    get_dev_data()
