import os
import sys

from cpath import data_path


def get_run_config():
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = os.path.join(data_path, "run_config", "cppnc_run.json")

    return path
