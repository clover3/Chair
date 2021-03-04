import sys
from collections import Counter, defaultdict

from cache import load_from_pickle
from misc_lib import group_by, get_first, get_second
from tab_print import print_table


def main():
    counter = load_from_pickle(sys.argv[1])

    n_appear = Counter()
    n_max = Counter()

    n_doc = 0
    for key, cnt in counter.items():
        max_idx, num_seg = key
        for i in range(num_seg):
            n_appear[i] += cnt

        n_max[max_idx] += cnt
        n_doc += cnt


    head = ["idx", "n appear", "n_max", "P(appear)", "P(max)", "P(max|appaer)"]
    rows = [head]
    for i in range(20):
        row = [i, n_appear[i], n_max[i], n_appear[i] / n_doc, n_max[i] / n_doc, n_max[i] / n_appear[i]]
        rows.append(row)
    print_table(rows)


def main2():
    counter = load_from_pickle(sys.argv[1])

    couter_per_length = defaultdict(Counter)

    l = []
    for key, cnt in counter.items():
        max_idx, num_seg = key
        e = num_seg, max_idx, cnt
        l.append(e)

    grouped = group_by(l, get_first)

    rows = []
    for num_seg in range(1, 20):
        entries = grouped[num_seg]
        cnt_sum = sum([cnt for _, max_idx, cnt in entries])

        local_counter = Counter()
        for _, max_idx, cnt in entries:
            local_counter[max_idx] = cnt


        row = [num_seg, cnt_sum]
        for seg_loc in range(num_seg):
            row.append(local_counter[seg_loc])
        rows.append(row)

    print_table(rows)



if __name__ == "__main__":
    main2()