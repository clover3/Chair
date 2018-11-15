
import csv
import os
from data_generator.common import *
from data_generator.common import _get_or_generate_vocab
corpus_dir = os.path.join(data_path, "wiki_lm")
train_path = os.path.join(corpus_dir, "enwiki_train.txt")

vocab_size = 32000
import random

class DataLoader():
    def __init__(self):
        self.train_data = None
        self.dev_data = None


    def example_generator(self, corpus_path):
        f = open(corpus_path, "r", encoding="utf-8", errors="ignore")
        reader = csv.reader(f, delimiter=',')

        for idx, row in enumerate(reader):
            if idx == 0: continue  # skip header
            # Works for both splits even though dev has some extra human labels.
            sent = row[0]
            yield {
                "inputs": sent,
            }

    def load_train_data(self):
        plain_data = self.example_generator(train_path)
        coded_data = list(self.encode(plain_data))
        random.seed(0)
        random.shuffle(coded_data)
        train_size = 0.9 * len(coded_data)
        dev_size = len(coded_data) - train_size
        self.train_data = coded_data[:train_size]
        self.dev_data = coded_data[train_size:]
