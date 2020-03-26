import os
import random
import shutil
import time
from collections import Counter
from time import gmtime, strftime



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
        if sample_size == 10 and self.total_repeat > 10000:
            sample_size = 100
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

class CodeTiming:
    def __init__(self):
        self.acc = {}
        self.prev_tick = {}


    def tick_begin(self, name):
        self.prev_tick[name] = time.time()

    def tick_end(self, name):
        elp = time.time() - self.prev_tick[name]
        if name not in self.acc:
            self.acc[name] = 0

        self.acc[name] += elp


    def print(self):
        for key in self.acc:
            print(key, self.acc[key])




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


def increment_circular(j, max_len):
    j += 1
    if j == max_len:
        j = 0
    return j


def pick1(l):
    return l[random.randrange(len(l))]


def pick2(l):
    return random.sample(l, 2)


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


# returns dictionary where key is the element in the iterable and the value is the func(key)


# returns dictionary where value is the func(value) of input dictionary


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


def get_dir_files(dir_path):
    path_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for filename in filenames:
            path_list.append(os.path.join(dir_path, filename))

    return path_list

def get_dir_files2(dir_path):
    r = []
    for item in os.scandir(dir_path):
        r.append(os.path.join(dir_path, item.name))

    return r

def get_dir_files_sorted_by_mtime(dir_path):
    path_list = get_dir_files(dir_path)
    path_list.sort(key=lambda x: os.path.getmtime(x))
    return path_list


def get_dir_dir(dir_path):
    path_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for dirname in dirnames:
            path_list.append(os.path.join(dir_path, dirname))

    return path_list


def sample_prob(prob):
    v = random.random()
    for n, p in prob:
        v -= p
        if v < 0:
            return n
    return 1


def list_print(l, width):
    cnt = 0
    for item in l:
        print(item, end=" / ")
        cnt += 1
        if cnt == width:
            print()
            cnt = 0
    print()


def group_by(interable, key_fn):
    grouped = {}
    for elem in interable:
        key = key_fn(elem)
        if key not in grouped:
            grouped[key] = list()

        grouped[key].append(elem)
    return grouped


def assign_list_if_not_exists(dict_like, key):
    if key not in dict_like:
        dict_like[key] = list()


def assign_default_if_not_exists(dict_like, key, default):
    if key not in dict_like:
        dict_like[key] = default()


class BinHistogram:
    def __init__(self, bin_fn):
        self.counter = Counter()
        self.bin_fn = bin_fn

    def add(self, v):
        self.counter[self.bin_fn(v)] += 1




class BinAverage:
    def __init__(self, bin_fn):
        self.list_dict = {}
        self.bin_fn = bin_fn

    def add(self, k, v):
        bin_id = self.bin_fn(k)
        if bin_id not in self.list_dict:
            self.list_dict[bin_id] = []

        self.list_dict[bin_id].append(v)

    def all_average(self):
        output = {}
        for k, v in self.list_dict.items():
            output[k] = average(v)
        return output


class DictValueAverage:
    def __init__(self):
        self.acc_dict = Counter()
        self.cnt_dict = Counter()

    def add(self, k, v):
        self.cnt_dict[k] += 1
        self.acc_dict[k] += v

    def avg(self, k):
        return self.acc_dict[k] / self.cnt_dict[k]

    def all_average(self):
        output = {}
        for k, v in self.cnt_dict.items():
            output[k] = self.avg(k)
        return output


class IntBinAverage(BinAverage):
    def __init__(self):
        super(IntBinAverage, self).__init__(lambda x: int(x))


def k_th_score(arr, k, reverse):
    return sorted(arr, reverse=reverse)[k]


def apply_threshold(arr, t):
    return [v  if v > t else 0 for v in arr]


def get_f1(prec, recall):
    if prec + recall != 0 :
        return (2 * prec * recall) / (prec + recall)
    else:
        return 0


def split_7_3(list_like):
    split = int(0.7 * len(list_like))

    train = list_like[:split]
    val = list_like[split:]
    return train, val


def file_iterator_interval(f, st, ed):
    for idx, line in enumerate(f):
        if idx < st:
            pass
        elif idx < ed:
            yield line
        else:
            break