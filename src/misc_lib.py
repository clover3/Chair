import time
import os
import shutil
from time import gmtime, strftime
import random

def average(l):
    return sum(l) / len(l)


def tprint(text):
    tim_str = strftime("%H:%M:%S", gmtime())
    print("{} : {}".format(tim_str, text))

class TimeEstimator:
    def __init__(self, total_repeat, name = "", sample_size = 10):
        self.time_analyzed = None
        self.time_count = 0
        self.total_repeat = total_repeat
        self.name = name
        self.base = 3
        self.sample_size = sample_size
        self.progress_tenth = 1

    def tick(self):
        self.time_count += 1
        if not self.time_analyzed:
            if self.time_count == self.base:
                self.time_begin = time.time()

            if self.time_count == self.base + self.sample_size:
                elapsed = time.time() - self.time_begin
                expected_sec = elapsed / self.sample_size * self.total_repeat
                expected_min = int(expected_sec / 60)
                print("Expected time for {} : {} min".format(self.name, expected_min))
                self.time_analyzed = True
        if self.total_repeat * self.progress_tenth / 10 < self.time_count:
            print("{}0% completed".format(self.progress_tenth))
            self.progress_tenth += 1

def exist_or_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def delete_if_exist(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def score_sort(list):
    return sorted(list, key = lambda x:-x[1])

def print_shape(name, tensor):
    print("{} shape : {}".format(name, tensor.shape))

def reverse_voca(word2idx):
    OOV = 1
    PADDING = 0
    idx2word = dict()
    idx2word[1] = "OOV"
    idx2word[0] = "PADDING"
    idx2word[3] = "LEX"
    for word, idx in word2idx.items():
        idx2word[idx] = word
    return idx2word

def slice_n_pad(seq, target_len, pad_id):
    coded_text = seq[:target_len]
    pad = (target_len - len(coded_text)) * [pad_id]
    return coded_text + pad



def get_textrizer(word2idx):
    idx2word = reverse_voca(word2idx)
    def textrize(indice):
        text = []
        PADDING = 0
        for i in range(len(indice)):
            word = idx2word[indice[i]]
            if word == PADDING:
                break
            text.append(word)
        return text
    return textrize

def get_textrizer_plain(word2idx):
    idx2word = reverse_voca(word2idx)
    def textrize(indice):
        text = []
        PADDING = 0
        for i in range(len(indice)):
            word = idx2word[indice[i]]
            if indice[i] == PADDING:
                break
            text.append(word)
        return " ".join(text)
    return textrize

def reverse(l):
    return list(reversed(l))

def flatten(z):
    return [y for x in z for y in x]

def left(pairs):
    return list([a for a,b in pairs])

def right(pairs):
    return list([b for a,b in pairs])


def increment_circular(j, max_len):
    j += 1
    if j == max_len:
        j = 0
    return j

def pick1(l):
    return l[random.randrange(len(l))]

