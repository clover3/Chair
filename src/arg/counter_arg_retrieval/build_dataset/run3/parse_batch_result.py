import os

from cpath import output_path
from misc_lib import get_dir_files


def main():
    dir_path = os.path.join(output_path,
                             "ca_building",
                             "run3",
                             "batch_result")
    for file_path in get_dir_files(dir_path):
        pass

    return NotImplemented


if __name__ == "__main__":
    main()