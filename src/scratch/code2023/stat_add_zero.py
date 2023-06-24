import random

import numpy as np
from scipy import stats


def binary_gen(n_sample, rate):
    items = []
    for _ in range(n_sample):
        if random.random() < rate:
            v = 1
        else:
            v = 0
        items.append(v)
    return items


def main():
    sample1 = binary_gen(100, 0.7)
    sample2 = binary_gen(100, 0.6)

    diff, p = stats.ttest_rel(sample1, sample2)
    print("Before", p)

    sample1 += [0] * 300
    sample2 += [0] * 300
    diff, p = stats.ttest_rel(sample1, sample2)
    print("After", p)



if __name__ == "__main__":
    main()