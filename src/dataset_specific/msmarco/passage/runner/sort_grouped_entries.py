from dataset_specific.msmarco.passage.grouped_reader import iter_train_from_partitioned_file10K, \
    get_train_query_grouped_dict_10K
from list_lib import foreach
from cpath import output_path
from misc_lib import path_join
from trainer_v2.chair_logging import c_log


def re_order_write_for_partition(save_root, partition_no):
    d = get_train_query_grouped_dict_10K(partition_no)
    save_path = path_join(save_root, str(partition_no))
    f = open(save_path, "w")

    def write_item(e):
        s = "\t".join(e)
        f.write(s.strip() + "\n")

    for qid, items in d.items():
        foreach(write_item, items)


def main():
    save_root = path_join("data", "msmarco", "passage", "group_sorted_10K")
    for i in range(110):
        c_log.info("Work for {}".format(i))
        try:
            re_order_write_for_partition(save_root, i)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()
