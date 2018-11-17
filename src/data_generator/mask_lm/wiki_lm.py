
import csv
import os
from data_generator.common import *
corpus_dir = os.path.join(data_path, "enwiki")
train_path = os.path.join(corpus_dir, "enwiki_train.txt")

vocab_size = 32000
import random

class DataLoader():
    def __init__(self):
        self.train_data = None
        self.dev_data = None


    def load_train_data(self):
        plain_data = NotImplemented
        coded_data = NotImplemented
        random.seed(0)
        random.shuffle(coded_data)
        train_size = 0.9 * len(coded_data)
        dev_size = len(coded_data) - train_size
        self.train_data = coded_data[:train_size]
        self.dev_data = coded_data[train_size:]
