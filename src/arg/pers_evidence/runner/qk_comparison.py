import numpy as np

from cache import load_from_pickle


def main():
    qk_list1 = load_from_pickle("perspective_qk_candidate_filtered_train")
    qk_list2 = load_from_pickle("pc_evi_filtered_qk_train")

    def print_qk(qk_list):
        l_list = list([len(k) for q, k in qk_list])
        avg = np.average(l_list)
        std = np.std(l_list)
        print("avg", avg)
        print("std", std)
        return avg, std

    print("PC")
    print_qk(qk_list1)
    print("Evidence")
    print_qk(qk_list2)


if __name__ == "__main__":
    main()