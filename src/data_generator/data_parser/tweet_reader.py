import os
from data_generator.common import *
from collections import defaultdict

topics = ["atheism", "climate", "abortion", "feminism", "hillary"]

corpus_dir = os.path.join(data_path, "tweets")

data_limit = 0
#data_limit = 30* 1000

def read_tsv(topic):
    path = os.path.join(corpus_dir, "{}.txt".format(topic))
    f = open(path, encoding="utf-8")
    cnt = 0
    for line in f:
        idx_tab = line.find("\t")
        if idx_tab <= 0:
            print("Wrong line ", line)
            continue
        id = line[:idx_tab]
        content = line[idx_tab+1:]
        cnt += 1
        if data_limit > 0 and cnt > data_limit:
            break
        yield id, content



def load_as_text_chunk(topic):
    collection = read_tsv(topic)
    for id, content in collection:
        yield content


def load_per_user(topic):
    cnt =00
    collection = read_tsv(topic)
    user_tweets = defaultdict(list)
    for id, content in collection:
        user_tweets[id].append(content)
        cnt += 1
        if data_limit > 0 and cnt > data_limit:
            break

    return user_tweets

