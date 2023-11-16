import pickle
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path
from misc_lib import path_join

from misc_lib import get_second
from trainer_v2.per_project.transparency.misc_common import save_tsv


def work(df_pickle_path, save_path):
    df: Dict[str, int] = pickle.load(open(df_pickle_path, "rb"))
    items = list(df.items())
    items.sort(key=get_second, reverse=True)
    save_tsv(items, save_path)


def main():
    work(
        path_join(output_path, "mmp", "bt2", "df.pickle"),
        path_join(output_path, "mmp", "bt2_df.tsv"),
    )



if __name__ == "__main__":
    main()