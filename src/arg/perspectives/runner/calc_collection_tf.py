from collections import Counter

from cache import load_from_pickle, save_to_pickle


def work():
    acc_counter = Counter()
    for i in range(0, 122):
        save_name = "acc_count_{}".format(i)
        counter = load_from_pickle(save_name)
        acc_counter.update(counter)
    save_to_pickle(acc_counter, "acc_count")

if __name__ == '__main__':
    work()
