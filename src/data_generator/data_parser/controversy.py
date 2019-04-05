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
        if rating < 2.5:
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


def load_pseudo_controversy_docs(name):
    dir_path = os.path.join(scope_dir, "pseudo_docs", name)
    return load_dir_docs(dir_path)


def load_dir_docs(dir_path):
    f = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        f.extend(filenames)
        break

    result = []
    for filename in f:
        file_path = os.path.join(dir_path, filename)
        lines = open(file_path, "r").readlines()
        if len(lines) < 1:
            print("Broken file : ", filename)
            continue
        doc_rank = int(filename.split(".")[0])
        doc_name = lines[0].split(":")[1].strip()

        tag_len = len("</TEXT>\n")
        content = lines[4]
        content = content[:-tag_len]
        result.append((doc_rank, doc_name, content))

    result.sort(key=get_first)
    return result


def load_tf(name):
    file_path = os.path.join(scope_dir, "pseudo_docs", name)
    return load_tf_inner(file_path)


def load_tf_inner(file_path):
    f = open(file_path, "r", encoding="utf-8")
    lines = f.readlines()
    tf_dict = Counter()
    ctf = int(lines[0])
    for line in lines[1:]:
        tokens = line.split()
        if len(tokens) != 3:
            continue

        word, tf, df = line.split()
        tf = int(tf)
        df = int(df)
        tf_dict[word] = tf

    return ctf, tf_dict


if __name__ == '__main__':
    cross_check()

