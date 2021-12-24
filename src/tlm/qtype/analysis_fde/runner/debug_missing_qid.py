import os
from collections import Counter

from cache import load_pickle_from
from cpath import output_path


def run_save_to_pickle_train():
    # MMD_train_qe_de_distill_base_prob
    #
    run_name = "qtype_2V_v_train_200000"
    split = "train"
    pred_path = os.path.join(output_path, "qtype", run_name + ".score")

    batches = load_pickle_from(pred_path)
    num_record = 0
    counter = Counter()
    for batch in batches:
        num_item = len(batch["data_id"])
        for i in range(num_item):
            data_id = batch["data_id"][i, 0]
            counter[data_id] += 1
        num_record += num_item

    print("Number of unique data id={}".format(len(counter)))
    print("Number of record {}".format(num_record))




def main():
    run_save_to_pickle_train()


if __name__ == "__main__":
    main()