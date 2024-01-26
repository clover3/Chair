import pickle, sys
from collections import Counter


def main():

    df_d = pickle.load(open(sys.argv[1], "rb"))
    c = Counter(df_d)

    for term, cnt in c.most_common(100000):
        print(term)


if __name__ == "__main__":
    main()