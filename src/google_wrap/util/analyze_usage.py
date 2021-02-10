import os
from collections import defaultdict

from google_wrap.get_storage_name import get_storage_name
from google_wrap.monitor_checkpoint import get_file_list
from misc_lib import get_second


def get_sub_dir_size(client, target_path):
    blobs = client.list_blobs(get_storage_name(), prefix=target_path)
    print(blobs)
    for blob in blobs:
        print(blob.size)


def main():
    os.environ["storage_name"] = "clover_eu4"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/youngwookim/clovertpu-483d14c880bf.json"
    root_path = "training/model/"
    dir_list = get_file_list(root_path)
    group = defaultdict(list)
    for dir_blob in dir_list:
        file_path = dir_blob['name']
        tokens = file_path.split("/")
        assert tokens[0] == "training"
        assert tokens[1] == "model"
        group[tokens[2]].append(dir_blob)

    summary = []
    for sub_dir, blobs in group.items():
        all_size = sum([b['size'] for b in blobs])
        summary.append((sub_dir, all_size))

    summary.sort(key=get_second, reverse=True)

    gb = 1024 * 1024 * 1024
    acc_all_size = 0
    for sub_dir, all_size in summary:
        ##
        # print("{0} {1:.2f}".format(sub_dir, all_size/gb))
        acc_all_size += all_size
    print("Total of {0:.2f} GB".format(acc_all_size/gb))




if __name__ == "__main__":
    main()