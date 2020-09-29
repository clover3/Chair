import sys

from clueweb.clueweb_galago_db import insert_ranked_list_from_path
from galagos.query_runs_ids import Q_CONFIG_ID_DEV_ALL
from galagos.types import *
from misc_lib import get_dir_files


def work(dir_path: FilePath):
    q_config_id = Q_CONFIG_ID_DEV_ALL
    print(dir_path)
    for file_path in get_dir_files(dir_path):
        print(file_path)
        ##
        insert_ranked_list_from_path(file_path, q_config_id)


if __name__ == "__main__":
    work(sys.argv[1])