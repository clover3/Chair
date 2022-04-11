from typing import List, Callable

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from cpath import at_output_dir
from datastore.sql_based_cache_client import SQLBasedCacheClientStr

FUNC_SIG = Callable[[List[SegmentedInstance]], List[float]]


def get_nli_cache_sqlite_path():
    return at_output_dir("sqlite_cache", "nli_cache.sqlite")


def main():
    test_save()


if __name__ == "__main__":
    main()
