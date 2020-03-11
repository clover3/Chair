import pickle
import sys
from collections import Counter

from misc_lib import TimeEstimator


def merge_counter(file_prefix, st, ed):
    count = Counter()
    ticker = TimeEstimator(ed-st)
    for i in range(st, ed):
        path = file_prefix + str(i)
        d = pickle.load(open(path, "rb"))

        for key in d:
            count[key] += d[key]
        ticker.tick()
    out_path = file_prefix + "_merged"
    pickle.dump(count, open(out_path, "wb"))



if __name__ == "__main__":
    merge_counter(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))