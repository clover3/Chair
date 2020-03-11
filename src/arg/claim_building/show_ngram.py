import sys

from arg.claim_building.count_ngram import load_n_gram_from_pickle, is_single_char_n_gram


def show(n):
    topic = "abortion"
    count = load_n_gram_from_pickle(topic, n)
    l = list(count.items())
    skip_count = 0
    l.sort(key=lambda x:x[1], reverse=True)
    for n_gram, cnt in l[:1000]:
        if is_single_char_n_gram(n_gram):
            skip_count += 1
        else:
            print(n_gram, cnt)

    print("Skip", skip_count)




if __name__ == "__main__":
    n = int(sys.argv[1])
    show(n)
