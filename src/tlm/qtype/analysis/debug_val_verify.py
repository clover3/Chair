import numpy as np

from cache import load_pickle_from


def main():
    obj = load_pickle_from("mmd_qtype_M")
    for batch in obj:
        print(batch["debug_val"])

        a = batch['qtype_weights_paired'][:, 0, :]
        b = batch['qtype_weights_paired'][:, 1, :]
        diff = a - b
        diff = np.sum(diff * diff)
        print("qtype_weights_diff", diff)


def main2():
    obj = load_pickle_from("mmd_qtype_M")
    n_node = 8
    for batch in obj:
        per_node = np.reshape(batch["debug_val"], [n_node, -1, 512])
        for j in range(n_node):
            concat_input_ids = per_node[j]
            q_only_and_full = np.reshape(concat_input_ids, [2, -1, 512])
            q_only = q_only_and_full[0]
            full_query = q_only_and_full[1]
            print("q_only", q_only[:, 30])
            print("full_query", full_query[:, 30])


if __name__ == "__main__":
    main()
