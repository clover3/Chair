from dataset_specific.msmarco.passage.passage_resource_loader import enum_queries
from misc_lib import path_join
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import save_tsv


def main():
    dev1000_query_path = path_join("data", "msmarco", "dev_sample1000", "queries.tsv")
    queries = list(tsv_iter(dev1000_query_path))
    assert len(queries) == 1000

    for i in range(10):
        st = i * 100
        ed = st + 100
        new_query_save_path = path_join("data", "msmarco", "dev_sample1000", f"queries_{i}.tsv")
        save_tsv(queries[st:ed], new_query_save_path)


if __name__ == "__main__":
    main()
