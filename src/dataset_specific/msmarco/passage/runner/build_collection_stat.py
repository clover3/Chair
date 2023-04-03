import itertools

from adhoc.kn_tokenizer import count_df
from cache import save_to_pickle
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_collection
from misc_lib import get_second, TELI


def main():
    itr = load_msmarco_collection()
    size = 8841823
    itr = TELI(itr, size)
    df = count_df(map(get_second, itr))
    save_to_pickle(df, "msmarco_passage_df")


if __name__ == "__main__":
    main()