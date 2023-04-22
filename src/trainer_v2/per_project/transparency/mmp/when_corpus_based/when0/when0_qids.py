from cache import save_to_pickle
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_sub_samples, tsv_iter
from cpath import output_path
from misc_lib import path_join


def main():
    itr = tsv_iter(path_join(output_path, "msmarco", "passage", "when_full", "0"))

    qid_set = set()
    for e in itr:
        qid = e[0]
        qid_set.add(qid)

    save_to_pickle(qid_set, "when0_qids")


if __name__ == "__main__":
    main()