from dataset_specific.msmarco.passage.passage_resource_loader import enum_queries
from trainer_v2.per_project.transparency.misc_common import save_tsv
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path
from misc_lib import path_join


def main():
    split = "dev"
    selected = []
    for qid, query in enum_queries(split):
        tokens = query.lower().split()
        if "car" in tokens:
            selected.append((qid, query))

    save_path = path_join(output_path, "mmp", "bias", f"car_queries_{split}.tsv")
    save_tsv(selected, save_path)



if __name__ == "__main__":
    main()