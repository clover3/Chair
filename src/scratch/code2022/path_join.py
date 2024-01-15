import os
from cpath import output_path
from misc_lib import path_join
from cpath import output_path
from misc_lib import path_join


def path_join_dev(*args):
    if len(args) > 1:
        parent_path = os.path.join(*args[:-1])
        print(parent_path)
        exist_or_mkdir(parent_path)
    return os.path.join(*args)


def main():
    path_join_dev(output_path, "something")


if __name__ == "__main__":
    main()