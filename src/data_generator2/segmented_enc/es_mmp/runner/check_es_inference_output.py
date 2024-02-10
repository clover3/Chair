import sys

from cache import load_pickle_from
from misc_lib import path_join, TimeEstimator


def do_for_partition(save_dir, partition_no):
    def get_save_path_fn(batch_idx):
        return path_join(save_dir, f"{partition_no}_{batch_idx}")

    data_size = 1000000
    save_batch_size = 1024 * 8
    num_batch = data_size // save_batch_size
    ticker = TimeEstimator(num_batch)
    n = 0
    try:
        for batch_idx in range(10000):
            save_path = get_save_path_fn(batch_idx)
            item = load_pickle_from(save_path)
            n += len(item)
            ticker.tick()
    except FileNotFoundError:
        pass

    print(f"total of {n} items")


if __name__ == "__main__":
    save_dir = sys.argv[1]
    job_no = int(sys.argv[2])
    do_for_partition(save_dir, job_no)
