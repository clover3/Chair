import xml.sax
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from path import *
import math
import os
from cache import *
import glob
import re

scope_dir = os.path.join(data_path, "controversy")
corpus_dir = os.path.join(scope_dir, "clueweb")
#corpus_dir = os.path.join(scope_dir, "web_hard")

def clean_text(text):
    return re.sub(r"<.*?>", " ", text)


def load_docs():
    docs_dir = os.path.join(corpus_dir, "docs")
    f = []
    for (dirpath, dirnames, filenames) in os.walk(docs_dir):
        f.extend(filenames)
        break

    result = []
    for filename in f:
        path = os.path.join(docs_dir, filename)
        raw_doc = open(path, "r").read()
        doc = clean_text(raw_doc)
        if not doc:
            print(filename)
            print(raw_doc)
        result.append((filename, doc))

    return result


def load_rating():
    rating_path = os.path.join(corpus_dir, "avg_rating.txt")
    f = open(rating_path, "r")
    for line in f:
        doc_id, rating = line.split()
        yield doc_id, float(rating)

def load_label():
    ratings = load_rating()

    labels = dict()
    for doc_id, rating in ratings:
        if rating <= 2:
            labels[doc_id] = 1
        else:
            labels[doc_id] = 0

    num_cont = sum(labels.values())
    return labels


def cross_check():
    labels = load_label()
    docs = load_docs()

    print(len(labels))
    print(len(docs))



    for doc_id, text in docs:
        print(doc_id, labels[doc_id], len(text))


if __name__ == '__main__':
    cross_check()

