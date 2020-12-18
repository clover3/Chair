import json
import os

from base_type import FilePath
from cpath import data_path
from misc_lib import get_dir_files, exist_or_mkdir


def main():
    dir_path = os.path.join(data_path, "pc_evi_qck_predict_dev_info")
    out_dir_path = os.path.join(data_path, "pc_evi_qck_predict_dev_info_fixed")
    exist_or_mkdir(out_dir_path)
    for file_path in get_dir_files(FilePath(dir_path)):
    # print(file_path)
    # file_path = "/mnt/nfs/work3/youngwookim/job_man/pc_evi_qck_predict_dev_info/0.info"
    # out_file_path = "/mnt/nfs/work3/youngwookim/job_man/temp_0.info"
        out_file_path = os.path.join(out_dir_path, os.path.basename(file_path))
        modify_and_save(file_path, out_file_path)


def modify_and_save(file_path, out_file_path):
    j = json.load(open(file_path, "r", encoding="utf-8"))
    for data_id, info in j.items():
        doc_id = info['candidate'][0]
        doc_part_id = "{}_{}".format(doc_id, 1)
        info['candidate'][0] = doc_part_id
    json.dump(j, open(out_file_path, "w"))


if __name__ == "__main__":
    main()


