import sys

from cpath import at_data_dir
from misc_lib import get_f1, NamedAverager


def load(file_path, cut):
    l = []
    for line in open(file_path, "r"):
        terms = line.split()
        terms = terms[:cut]
        l.append(terms)
    return l


def main():
    data_name = sys.argv[1]
    gold = load(at_data_dir("genex", "{}_gold.txt".format(data_name)), 999)
    run1 = load(sys.argv[2], 3)

    def common(pred, gold):
        return list([t for t in pred if t in gold])

    d1 = NamedAverager()

    for idx, (t1, t_gold) in enumerate(zip(run1, gold)):
        c1 = common(t1, t_gold)
        p1 = len(c1) / len(t1)
        r1 = len(c1) / len(t_gold)
        f1 = get_f1(p1, r1)
        d1['prec'].append(p1)
        d1['recall'].append(r1)
        d1['f1'].append(f1)

    print(d1.get_average_dict())

if __name__ == "__main__":
    main()