import os
import sys

from tf_v2_support import tf_record_enum


def file_cnt(fn):
    c = 0
    for record in tf_record_enum(fn):
       c += 1
    return c


def count_instance(dir_name):
    acc = 0
    for _,_,filenames in os.walk(dir_name):
        for fn in filenames:
            path = os.path.join(dir_name, fn)
            cnt = file_cnt(path)
            print(fn, cnt)
            acc += cnt
    return acc


def count_print_as_d(dir_name):
    acc = 0
    d = {}
    for _,_,filenames in os.walk(dir_name):
        for fn in filenames:
            path = os.path.join(dir_name, fn)
            cnt = file_cnt(path)
            print(fn, cnt)
            d[fn] = cnt
            acc += cnt
    print(d)
    print("Total\t", acc)
    return acc

if __name__ == "__main__":
    p = sys.argv[1]
    if os.path.isdir(p):
        if len(sys.argv) > 2 and sys.argv[2] == "-d":
            count_print_as_d(p)
        else:
            print(count_instance(p))
    else:
        print("Number of record : ", file_cnt(p))
