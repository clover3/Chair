import pickle
from misc_lib import TimeEstimator, tprint
from collections import Counter


def summarize_stat(counter):
    max_len = max(counter.keys())
    total = sum(counter.values())
    for i in range(0, max_len, 10):
        local_sum = 0
        for j in range(i, i+10):
            local_sum += counter[j]

        print("{} ~ {} : {}".format(i, i+10, local_sum/total))


def load_title_from_pickle():
    return pickle.load(open("titles", "rb"))


def load_title():
    f = open("/mnt/nfs/work3/youngwookim/data/wiki_title/enwiki-20190720-all-titles","r")

    s = set()
    first_line = True
    ticker = TimeEstimator(48213789, "Task", 1000)

    c = Counter()
    for line in f:
        if first_line:
            first_line=False
            continue

        try:
            a, b= line.split()
            title = b.strip()
            s.add(title)
        except ValueError as e:
            print(line)



        ticker.tick()
    return s

long_cut = 20
def long_title_sig(titles):
    long_titles = set()
    sig_set = set()

    for t in titles:
        if len(t) > 20:
            sig = t[:20]
            sig_set.add(sig)
            long_titles.add(t)
    return long_titles, sig_set

def is_ending_dot(line):
    if line[-1] == ".":
        return True
    if line[-1] == " " and line[-2] == ".":
        return True
    else:
        return False

def is_ending_dot_cr(line):
    if line[-2] == ".":
        return True
    if line[-2] == " " and line[-3] == ".":
        return True
    else:
        return False

def dot_ending_sig(titles, sig_fn):
    de_titles = set()
    sig_set = set()

    for t in titles:
        if len(t) > 20:
            sig = sig_fn(t)
            sig_set.add(sig)
            de_titles.add(t)
    return de_titles, sig_set


class LengthSet:
    def __init__(self, s):
        self.d = {}

        for item in s:
            l = len(item)
            if l not in self.d:
                self.d[l] = set()
            self.d[l].add(item)

    def __contains__(self, item):
        l = len(item)
        if l in self.d:
            return item in self.d[l]
        else:
            return False


class WikiSpliter:
    def __init__(self):
        self.titles = load_title_from_pickle()
        tprint("titles : {}".format(len(self.titles)))
        self.long_titles, self.long_sig = long_title_sig(self.titles)
        tprint("Long Titles : {}".format(len(self.long_titles)))
        self.dot_end_title, self.de_sig = dot_ending_sig(self.titles, self.hash)
        tprint("Dot Ending Signature : {}".format(len(self.dot_end_title)))
        self.out_f = open("/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_train_bordered.txt", "w")

    def hash(self, line):
        return line[1] + line[5] + line[9] + line[13]

    def title_format(self, line):
        return line.strip().replace(" ", "_")

    def is_title(self, line):
        if len(line) > 50:
            return False
        elif len(line) < 20:
            if self.title_format(line) in self.titles:
                return True
            else:
                return False
        else:
            if is_ending_dot_cr(line):
                sig = self.hash(line)
                if sig in self.de_sig:
                    if self.title_format(line) in self.dot_end_title:
                        return True
            else:
                if self.title_format(line) in self.long_titles:
                    return True
        return False



    def flush(self, title, content):
        self.out_f.write(title + "\n")
        for line in content:
            self.out_f.write(line)
        self.out_f.write("EOD!\n")

    def parse(self, lines):

        content = []
        cur_title = None
        step = 126766466
        ticker = TimeEstimator(step, "reader", 200)
        for line in lines:
            is_title = False
            if cur_title is None:
                is_title = True
            elif not content:
                is_title = False
            else:
                is_title = self.is_title(line)

            if is_title:
                if cur_title is not None:
                    self.flush(cur_title, content)
                    content = []
                cur_title = line.strip()
            else:
                content.append(line)
            ticker.tick()




def parse():
    ws = WikiSpliter()
    f = open("/mnt/nfs/work3/youngwookim/data/enwiki4bert/enwiki_train.txt", "r")
    ws.parse(f)


def save_title_pickle():
    t = load_title()
    #pickle.dump(t, open("titles", "wb"))


if __name__ == "__main__":
    parse()
