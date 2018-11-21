
import os
import pickle
from data_generator.common import *
corpus_dir = os.path.join(data_path, "guardian")
import re

topics = ["atheism"]

def load_pickle(topic):
    pickle_path = os.path.join(corpus_dir, "{}.pickle".format(topic))
    collection = pickle.load(open(pickle_path, "rb"))

    return collection

def clean_text(text):
    return re.sub(r"<.*?>", "", text)

def load_as_text_chunk(topic):
    collection = load_pickle(topic)
    for dicussion_id, url, comments in collection:
        for comment_id, content, target_id in comments:
            yield clean_text(content)

