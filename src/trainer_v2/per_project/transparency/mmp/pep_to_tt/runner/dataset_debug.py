import sys

from misc_lib import get_dir_files
from table_lib import tsv_iter


def get_pep_tt_dataset(dir_path):
    file_list = get_dir_files(dir_path)
    q_set = set()
    d_pos_set = set()
    d_neg_set = set()
    triplet_set = set()
    d_pos_d = {}
    g_idx = 0
    for file_path in file_list:
        print(file_path)
        raw_train_iter = tsv_iter(file_path)
        for (q, d_pos, d_neg) in raw_train_iter:
            triplet = (q, d_pos, d_neg)

            if triplet in triplet_set:
                print("Triplet seen")
            else:
                triplet_set.add(triplet)
            if q in q_set:
                print(g_idx, "Query seen:")
            else:
                q_set.add(q)

            if d_pos in d_pos_set:
                print(g_idx, "d_pos seen")
            else:
                d_pos_d[d_pos] = g_idx
                d_pos_set.add(d_pos)

            # if d_neg in d_neg_set:
            #     print(g_idx, "d_neg seen")
            # else:
            #     d_neg_set.add(d_pos)
            g_idx += 1


def main():
    get_pep_tt_dataset(sys.argv[1])
    return NotImplemented


if __name__ == "__main__":
    main()
