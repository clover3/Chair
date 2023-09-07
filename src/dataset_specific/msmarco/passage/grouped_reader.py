from collections import defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set
from table_lib import tsv_iter
from misc_lib import path_join
from trainer_v2.chair_logging import c_log
# group_no: from 0 to 108 (inc)


# takes 40 secs
def get_train_query_grouped_dict_10K(group_no) -> Dict[str, List]:
    itr = iter_train_from_partitioned_file10K(group_no)
    return make_grouped(itr)


def make_grouped(itr):
    per_query_dict = defaultdict(list)
    for e in itr:
        qid, pid, _, _ = e
        l = per_query_dict[qid]
        l.append(e)
    return per_query_dict



def iter_train_from_partitioned_file10K(partition_no):
    grouping_root = path_join("data", "msmarco", "passage", "grouped_10K")
    file_path = path_join(grouping_root, str(partition_no))
    itr = tsv_iter(file_path)
    return itr


def main():
    c_log.info("Start ")
    get_train_query_grouped_dict_10K(0)
    c_log.info("Done")


if __name__ == "__main__":
    main()