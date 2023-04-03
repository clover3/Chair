from collections import defaultdict

from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter
from misc_lib import path_join


# group_no: from 0 to 108 (inc)
from trainer_v2.chair_logging import c_log


# takes 40 secs
def get_query_group_10K(group_no):
    itr = iter_train_query_grouped_10K(group_no)
    per_query_dict = defaultdict(list)
    for e in itr:
        qid, pid, _, _ = e
        l = per_query_dict[qid]
        l.append(e)

    return per_query_dict


def iter_train_query_grouped_10K(group_no):
    grouping_root = path_join("data", "msmarco", "passage", "grouped_10K")
    file_path = path_join(grouping_root, str(group_no))
    itr = tsv_iter(file_path)
    return itr


def main():
    c_log.info("Start ")
    get_query_group_10K(0)
    c_log.info("Done")


if __name__ == "__main__":
    main()