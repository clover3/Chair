import csv
import os
from data_generator.common import *
from data_generator.text_encoder import SubwordTextEncoder
import random

vocab_filename = "shared_voca.txt"

corpus_dir = os.path.join(data_path, "stance_detection")
vocab_size = 32000
num_classes = 3

stance_label = ["NONE", "AGAINST", "FAVOR"]

def get_train_text():
    corpus_path = os.path.join(corpus_dir, "train.csv")
    f = open(corpus_path, "r", encoding="utf-8", errors="ignore")
    reader = csv.reader(f, delimiter=',')

    for idx, row in enumerate(reader):
        if idx == 0: continue  # skip header
        # Works for both splits even though dev has some extra human labels.
        sent = row[0]
        yield sent


class DataLoader:
    def __init__(self):
        self.train_data = None
        self.dev_data = None
        self.test_data = None


        voca_path = os.path.join(data_path, vocab_filename)
        assert os.path.exists(voca_path)
        self.encoder = SubwordTextEncoder(voca_path)
        self.max_sequence = 140

    def example_generator(self, corpus_path, select_target):
        label_list = stance_label
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
        plain_data = list(self.example_generator(path, "Atheism"))
        random.seed(0)
        random.shuffle(plain_data)

        train_size = int(0.9 * len(plain_data))
        dev_size = len(plain_data) - train_size
        self.train_data_raw = plain_data[:train_size]
        self.dev_data_raw = plain_data[train_size:]

        self.train_data = self.encode(self.train_data_raw)
        self.dev_data = self.encode(self.dev_data_raw)

    def load_test_data(self):
        path = os.path.join(corpus_dir, "test.csv")
        self.test_data_raw = list(self.example_generator(path, "Atheism"))
        self.test_data = self.encode(self.test_data_raw)

    @classmethod
    def dict2tuple(cls, data):
        X = []
        Y = []
        for entry in data:
            X.append(entry["inputs"])
            Y.append(entry["label"])

        return X, Y

    def get_train_data(self):
        if self.train_data is None:
            self.load_train_data()

        return self.dict2tuple(self.train_data)

    def get_dev_data(self):
        if self.dev_data is None:
            self.load_train_data()

        return self.dict2tuple(self.dev_data)

    def get_test_data(self):
        if self.test_data is None:
            self.load_test_data()

        return self.dict2tuple(self.test_data)

    def encode(self, plain_data):
        for entry in plain_data:
            key = "inputs"
            coded_text = self.encoder.encode(entry[key])
            pad = (self.max_sequence - len(coded_text)) * [text_encoder.PAD_ID]
            entry[key] = coded_text + pad
            yield entry

