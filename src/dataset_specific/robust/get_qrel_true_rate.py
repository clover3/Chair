import os

from data_generator.data_parser.robust2 import robust_path
from trec.true_rate_count import count_true_rate


def main():
    path = os.path.join(robust_path, "qrels.rob04.txt")

    tru_cnt, neg_cnt = count_true_rate(path)

    print('true/neg', tru_cnt ,neg_cnt)
    print('rate', tru_cnt / (tru_cnt + neg_cnt))


if __name__ == "__main__":
    main()