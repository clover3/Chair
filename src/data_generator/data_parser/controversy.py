import re

from cache import *
from galagos import galago
from misc_lib import *

scope_dir = os.path.join(data_path, "controversy")
clueweb_dir = os.path.join(scope_dir, "clueweb")
#corpus_dir = os.path.join(scope_dir, "web_hard")

def clean_text(text):
    return re.sub(r"<.*?>", " ", text)


def load_clue303_docs():
    docs_dir = os.path.join(clueweb_dir, "docs")
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
    rating_path = os.path.join(clueweb_dir, "avg_rating.txt")
    f = open(rating_path, "r")
    for line in f:
        doc_id, rating = line.split()
        yield doc_id, float(rating)

def load_clue303_label():
    ratings = load_rating()

    labels = dict()
    for doc_id, rating in ratings:
        if rating < 2.5:
            labels[doc_id] = 1
        else:
            labels[doc_id] = 0

    return labels


def load_clueweb_testset():
    labels = load_clue303_label()
    docs = load_clue303_docs()

    dev_X = []
    dev_Y = []
    for name, doc in docs:
        dev_X.append(doc)
        dev_Y.append(labels[name])
    return dev_X, dev_Y


def cross_check():
    labels = load_clue303_label()
    docs = load_clue303_docs()

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
            #print("Broken file : ", filename)
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
    return galago.load_tf(file_path)


def load_guardian():
    dir_path = os.path.join(scope_dir, "guardian")
    todo = [
        ("guardianC.txt", 1) , ("guardianNC.txt",0)
    ]
    X = []
    Y = []
    for name, label in todo:
        path = os.path.join(dir_path, name)
        docs = open(path, "r").readlines()
        X.extend(docs)
        Y.extend([label for _ in docs])
    return X, Y

def load_guardian16_signal():
    path = os.path.join(scope_dir, "LM_train_docs.pickle")
    return pickle.load(open(path, "rb"))

def load_guardian_selective_signal():
    path = os.path.join(data_path, "guardian",  "LM_docs.pickle")
    return pickle.load(open(path, "rb"))


if __name__ == '__main__':
    cross_check()

