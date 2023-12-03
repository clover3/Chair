from dataset_specific.msmarco.passage.passage_resource_loader import enum_queries
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import save_tsv
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path
from misc_lib import path_join


def main():
    split = "dev"
    query_path = path_join(output_path, "mmp", "bias", "car_exp", f"car_queries_{split}.tsv")
    qid_query_list = list(tsv_iter(query_path))

    term_list_path = path_join(output_path, "mmp", "bias", "car_exp", "car_maker_list.txt")
    term_list = [line.lower().strip() for line in open(term_list_path, "r")]
    target_terms = set(term_list)

    selected = []
    for qid, query in qid_query_list:
        tokens = query.lower().split()
        skip = False
        for t in tokens:
            if t in target_terms:
                skip = True
                break

        if not skip:
            selected.append((qid, query))

    save_path = path_join(output_path, "mmp", "bias", "car_exp", "car_queries_no_maker.tsv")
    save_tsv(selected, save_path)



if __name__ == "__main__":
    main()