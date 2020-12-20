from cache import load_from_pickle


def main():
    qk_list1 = load_from_pickle("perspective_qk_candidate_filtered_train")
    qk_list2 = load_from_pickle("pc_evi_filtered_qk_train")


    def print_qk(qk_list):
        print("----")
        for q, k in qk_list[:30]:
            print(len(k), end=" ")
        print("")

    print_qk(qk_list1)
    print_qk(qk_list2)



if __name__ == "__main__":
    main()