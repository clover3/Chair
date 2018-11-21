import os
from data_generator.common import *

topics = ["atheism", "climate", "abortion", "feminism"]

corpus_dir = os.path.join(data_path, "tweets")

def read_tsv(topic):
    path = os.path.join(corpus_dir, "{}.txt".format(topic))
    f = open(path, encoding="utf-8")
    for line in f:
        idx_tab = line.find("\t")
        assert idx_tab > 0
        id = line[:idx_tab]
        content = line[idx_tab+1:]
        yield id, content


def load_as_text_chunk(topic):
    collection = read_tsv(topic)
    for id, content in collection:
        yield content

