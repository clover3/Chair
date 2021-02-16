import os

from cpath import data_path
from list_lib import left
from misc_lib import exist_or_mkdir
from tlm.robust.load import robust_query_intervals


def main():
    interval_start_list = left(robust_query_intervals)
    root_dir = os.path.join(data_path, "robust_split")
    exist_or_mkdir(root_dir)

    def save_to_file(file_name, s):
        save_path = os.path.join(root_dir, file_name)
        open(save_path, "w").write(s)

    for split_idx in range(5):
        held_out = interval_start_list[split_idx]
        train_items = interval_start_list[:split_idx] + interval_start_list[split_idx+1:]

        save_to_file("{}_{}".format(str(split_idx), "train"), " ".join(map(str, train_items)))
        save_to_file("{}_{}".format(str(split_idx), "test"), str(held_out))


if __name__ == "__main__":
    main()