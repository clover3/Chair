import json
import os

from base_type import FilePath
from epath import job_man_dir
from misc_lib import get_dir_files


def main():
    dir_path = os.path.join(job_man_dir, "qcknc_dense_val_info")
    out_dir_path = os.path.join(job_man_dir, "qcknc_dense_val_info_light")
    for file_path in get_dir_files(FilePath(dir_path)):
        print(file_path)
        if file_path.endswith(".info"):
            out_file_path = os.path.join(out_dir_path, os.path.basename(file_path))
            j = json.load(open(file_path, "r", encoding="utf-8"))
            for data_id, info in j.items():
                info['kdp'][3] = []
            json.dump(j, open(out_file_path, "w"))


if __name__ == "__main__":
    main()


