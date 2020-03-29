import sys

from arg.perspectives.clueweb_galago_db import insert_ranked_list_from_path
from arg.perspectives.ranked_list_interface import Q_CONFIG_ID_BM25_10000
from galagos.types import *
from misc_lib import get_dir_files


def work(dir_path: FilePath):
    q_config_id = Q_CONFIG_ID_BM25_10000
    print(dir_path)
    for file_path in get_dir_files(dir_path):
        print(file_path)
        insert_ranked_list_from_path(file_path, q_config_id)


if __name__ == "__main__":
    work(sys.argv[1])