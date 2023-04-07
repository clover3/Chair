import os

from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.data_enum import enum_pos_neg_sample, check_ranked_list_size


def main():
    itr = check_ranked_list_size(range(109))
    for t in itr:
        pass


if __name__ == "__main__":
    main()