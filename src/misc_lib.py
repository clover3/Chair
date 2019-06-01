import time
import os
import shutil
from time import gmtime, strftime
import random

def average(l):
    if len(l) == 0:
        return 0
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

def pair_shuffle(l):
    new_l = []
    for idx in range(0, len(l), 2):
        new_l.append( (l[idx], l[idx+1]) )

    random.shuffle(new_l)
    result = []
    for a,b in new_l:
        result.append(a)
        result.append(b)
    return result

class MovingWindow:
    def __init__(self, window_size):
        self.window_size = window_size
        self.history = []

    def append(self, average, n_item):
        all_span = self.history + [average] * n_item
        self.history = all_span[-self.window_size:]

    def append_list(self, value_n_item_list):
        for avg_val, n_item in value_n_item_list:
            self.append(avg_val, n_item)

    def get_average(self):
        if not self.history:
            return 0
        else:
            return average(self.history)


def get_first(x):
    return x[0]


def get_second(x):
    return x[1]


class OpTime:


    def time_op(self, fn):
        begin = time.time()
        ret = fn()

def lmap(func, iterable_something):
    return list([func(e) for e in iterable_something])


def flat_apply_stack(list_fn, list_of_list, verbose=True):
    item_loc = []

    flat_items = []
    for idx1, l in enumerate(list_of_list):
        for idx2, item in enumerate(l):
            flat_items.append(item)
            item_loc.append(idx1)

    if verbose:
        print("Total of {} items".format(len(flat_items)))
    results = list_fn(flat_items)

    assert len(results) == len(flat_items)

    stack = []
    cur_list = []
    line_no = 0
    for idx, item in enumerate(results):
        while idx >=0 and item_loc[idx] != line_no:
            assert len(cur_list) == len(list_of_list[line_no])
            stack.append(cur_list)
            line_no += 1
            cur_list = []
        cur_list.append(item)
    stack.append(cur_list)
    return stack



def parallel_run(input_list, list_fn, split_n):
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    from pathos.multiprocessing import ProcessingPool as Pool
    p = Pool(split_n, daemon=True)
    args = chunks(input_list, split_n)
    result_list_list = p.map(list_fn, args)

    result = []
    for result_list in result_list_list:
        result.extend(result_list)
    return result

def dict_reverse(d):
    inv_map = {v: k for k, v in d.items()}
    return inv_map