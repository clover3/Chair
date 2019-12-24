import os
import sys
import time


def get_dir_files(dir_path):
    path_list = []
    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        for filename in filenames:
            path_list.append(os.path.join(dir_path, filename))

    return path_list

def get_dir_files_sorted_by_mtime(dir_path):
    path_list = get_dir_files(dir_path)
    path_list.sort(key=lambda x: os.path.getmtime(x))
    return path_list


def time_since_modified(target_dir):
    files = get_dir_files_sorted_by_mtime(target_dir)
    last_modified_file_path = files[-1]
    time_diff = time.time() - os.path.getmtime(last_modified_file_path)
    return time_diff


if __name__ == "__main__":
    target_dir = sys.argv[1]
    while time_since_modified(target_dir) < 100:
        time.sleep(100)





