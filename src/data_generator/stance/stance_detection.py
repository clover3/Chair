import csv
import os
from data_generator.common import *
from data_generator.common import _get_or_generate_vocab
from data_generator.text_encoder import SubwordTextEncoder
import random


corpus_dir = os.path.join(data_path, "stance_detection")
vocab_size = 32000


class DataLoader:
    def __init__(self):
        self.train_data = None
        self.dev_data = None
        self.encoder = SubwordTextEncoder()

    def class_labels(self):
        return ["NONE", "AGAINST", "FAVOR"]

    def example_generator(self, corpus_path, select_target):
        label_list = self.class_labels()
        f = open(corpus_path, "r", encoding="utf-8", errors="ignore")
        reader = csv.reader(f, delimiter=',')

        for idx, row in enumerate(reader):
            if idx == 0: continue  # skip header
            # Works for both splits even though dev has some extra human labels.
            sent = row[0]
            target = row[1]
            label = label_list.index(row[2])
            if select_target is None:
                f_include = True
            else:
                if target in select_target:
                    f_include = True
                else:
                    f_include = False
            if f_include:
                yield {
                    "inputs": sent,
                    "label": label
                }


    def load_train_data(self):
        path = os.path.join(corpus_dir, "train.csv")
        plain_data = self.example_generator(path, "atheism")
        coded_data = list(self.encode(plain_data))
        random.seed(0)
        random.shuffle(coded_data)
        train_size = 0.9 * len(coded_data)
        dev_size = len(coded_data) - train_size
        self.train_data = coded_data[:train_size]
        self.dev_data = coded_data[train_size:]

    def train_data(self):
        if self.dev_data is None:
            self.load_train_data()
        return self.train_data

    def dev_data(self):
        if self.dev_data is None:
            self.load_train_data()
        return self.dev_data

    def encode(self, plain_data):
        tmp_dir = os.path.join(corpus_dir, "temp")
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        #symbolizer_vocab = _get_or_generate_vocab(
#            corpus_dir, 'vocab.subword_text_encoder', vocab_size)

        for entry in plain_data:
            key = "inputs"
            entry[key] = self.encoder.encode(entry[key])
            yield entry

