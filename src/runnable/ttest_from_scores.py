import sys

from scipy.stats import stats

from misc_lib import average


def read(path):
    d = {}
    for line in open(path, "r"):
        key, score = line.split("\t")
        score = float(score)
        d[key] = score
    return d


def main():
    path1 = sys.argv[1]
    path2 = sys.argv[2]

    score_d1 = read(path1)
    score_d2 = read(path2)
    # print
    pairs = []
    for key in score_d1:
        try:
            e = (score_d1[key], score_d2[key])
            pairs.append(e)
        except KeyError as e:
            pass

    if len(pairs) < len(score_d1) or len(pairs) < len(score_d2):
        print("{} matched from {} and {} scores".format(len(pairs), len(score_d1), len(score_d2)))

    l1, l2 = zip(*pairs)
    print(l1)
    print(l2)
    d, p_value = stats.ttest_rel(l1, l2)
    print("baseline:", average(l1))
    print("treatment:", average(l2))
    print(d, p_value)


if __name__ == "__main__":
    main()