from dataset_specific.msmarco.passage.path_helper import get_train_triples_small_path
from table_lib import tsv_iter


def main():
    itr = tsv_iter(get_train_triples_small_path())

    for q, dp, dn in itr:
        if "light" in q and "theory" in dp:
            print("Query: ", q)
            print("Pos Doc: ", dp)
            print("Neg Doc: ", dn)
            print()


if __name__ == "__main__":
    main()