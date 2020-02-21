import os
import pickle
import time

from data_generator.job_runner import sydney_working_dir
from misc_lib import get_dir_files
from tlm.data_gen.run_clueweb_tokenize import load_undone_file_list


def enum_directory1(dir_path):
    consecutive_not_exists = 0
    file_list = []
    for i in range(30000):
        file_path = os.path.join(dir_path, "{}.txt".format(i))

        if os.path.exists(file_path):
            file_list.append(file_path)
            consecutive_not_exists = 0
        else:
            consecutive_not_exists += 1

        if consecutive_not_exists > 20:
            break
    return file_list


def enum_directory2(dir_path):
    r = []
    for item in os.scandir(dir_path):
        r.append(os.path.join(dir_path, item.name))

    return r


def enum_directory3(dir_path):
    return get_dir_files(dir_path)


def job():
    print("1 load_undone_file_list")
    file_list = load_undone_file_list()
    # read marks
    def load_mark_list(dir_path, max_mark):
        l = []
        for i in range(max_mark):
            e = os.path.exists(os.path.join(dir_path, str(i)))
            if e :
                l.append(i)
        return l

    print("2 load_undone_file_list")
    mark_list = load_mark_list(os.path.join(sydney_working_dir, "clueweb_tokenize"), 8000)
    # refine file_list
    target_list = []
    for i, dir_path in enumerate(file_list):
        if i in mark_list :
            pass
        else:
            target_list.append(dir_path)

    dst_dir = "/dev/shm/clueweb12-13-text"

    print("3 for file_path in target_list:")
    for dir_path in target_list:
        st = time.time()
        _, dir_name = os.path.split(dir_path)
        r = enum_directory2(dir_path)
        raw_data = []
        for file_path in r:
            _, file_name = os.path.split(file_path)
            f = open(file_path, "r", errors="ignore")
            lines = f.readlines()
            doc_name = "{}_{}".format(file_name, dir_name)
            raw_data.append((doc_name, lines))

        out_path = os.path.join(dst_dir, doc_name)
        pickle.dump(raw_data, open(out_path, "wb"))
        ed = time.time()
        print("Elapsed : ", ed - st)

    #  Read directory, save as pickle


if __name__ == "__main__":
    job()
