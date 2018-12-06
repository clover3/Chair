import os
from data_generator.common import *
from collections import defaultdict
import re
import time

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

def is_english(text):
    alnum_cnt =0
    non_space = 0
    def is_roman(c):
        return c.isupper() or c.islower()
    for c in text:
        if is_roman(c):
            alnum_cnt += 1
        if not c.isspace():
            non_space += 1
    score = alnum_cnt / len(text)
    return score > 0.5



def load_per_user(topic):
    begin = time.time()
    cnt = 0
    collection = read_tsv(topic)
    user_tweets_raw = defaultdict(set)
    for id, content in collection:
        user_tweets_raw[id].add(content)
        cnt += 1
        if data_limit > 0 and cnt > data_limit:
            break

    user_tweets = defaultdict(list)
    for key, values in user_tweets_raw.items():
        l = list(values)
        if is_english(l[0]):
            user_tweets[key] = l

    elps = time.time() - begin
    print("Lang filter {} -> {}".format(len(user_tweets_raw), len(user_tweets)))
    print("load_per_user() : {} elapsed".format(elps))
    return user_tweets


def extract_mention(sent):
    reg = r"@[0-9a-zA-Z_]{1,15}"
    m = re.findall(reg, sent)

    user_ids = []
    for item in m:
        user_ids.append(item[1:])
    return user_ids
