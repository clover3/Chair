from dataset_specific.msmarco.passage.passage_resource_loader import enum_queries
from misc_lib import path_join
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import save_tsv


def main():
    dev1000_query_path = path_join("data", "msmarco", "dev_sample1000", "queries.tsv")
    dev1000_qids = [qid for qid, query in tsv_iter(dev1000_query_path)]
    new_query_save_path = path_join("data", "msmarco", "dev_sample1000_B", "queries.tsv")

    selected = []
    for qid, query in enum_queries("dev"):
        if qid not in dev1000_qids:
            selected.append((qid, query))

    save_tsv(selected, new_query_save_path)


if __name__ == "__main__":
    main()
