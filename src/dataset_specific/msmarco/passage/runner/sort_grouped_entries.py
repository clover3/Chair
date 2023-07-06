from dataset_specific.msmarco.passage.grouped_reader import iter_train_from_partitioned_file10K, \
    get_train_query_grouped_dict_10K, make_grouped
from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter
from list_lib import foreach
from cpath import output_path
from misc_lib import path_join
from trainer_v2.chair_logging import c_log


def write_groups_to(d, save_path):
    f = open(save_path, "w")

    def write_item(e):
        s = "\t".join(e)
        f.write(s.strip() + "\n")

    for qid, items in d.items():
        foreach(write_item, items)


def re_order_write_for_train(save_root, partition_no):
    grouping_root = path_join("data", "msmarco", "passage", "grouped_10K")
    src_path = path_join(grouping_root, str(partition_no))
    save_path = path_join(save_root, str(partition_no))

    itr = tsv_iter(src_path)
    per_query_dict = make_grouped(itr)
    write_groups_to(per_query_dict, save_path)


def re_order_write_for_split(split, save_root, partition_no):
    grouping_root = path_join("data", "msmarco", "passage", f"{split}_grouped_10K")
    src_path = path_join(grouping_root, str(partition_no))
    save_path = path_join(save_root, str(partition_no))

    itr = tsv_iter(src_path)
    per_query_dict = make_grouped(itr)
    write_groups_to(per_query_dict, save_path)


def main():
    save_root = path_join("data", "msmarco", "passage", "group_sorted_10K")
    for i in range(110, 120):
        c_log.info("Work for {}".format(i))
        try:
            re_order_write_for_train(save_root, i)
        except FileNotFoundError as e:
            print(e)


def main_for_dev():
    split = "dev"
    save_root = path_join("data", "msmarco", "passage", f"{split}_group_sorted_10K")
    for i in range(0, 120):
        c_log.info("Work for {}".format(i))
        try:
            re_order_write_for_split(split, save_root, i)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main_for_dev()
