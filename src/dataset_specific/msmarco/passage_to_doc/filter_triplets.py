from typing import List, Tuple

from cache import load_from_pickle, save_to_pickle
from dataset_specific.msmarco.passage.passage_resource_loader import enumerate_triple
from galagos.types import QueryID
from misc_lib import TimeEstimator


def main():
    PassageID = str
    positive_passage_list: List[Tuple[QueryID, PassageID]] = load_from_pickle("msmarco_doc_joined_passage_list")
    positive_passage_set = set(positive_passage_list)
    output = []
    n_record = 397768673
    ticker = TimeEstimator(n_record, "enum", 100000)

    for qid, pid1, pid2 in enumerate_triple():
        ticker.tick()
        key = qid, pid1
        if key in positive_passage_set:
            positive_passage_set.remove(key)
            output.append((qid, pid1, pid2))
    save_to_pickle(output, "msmarco_doc_joined_passage_triples")

    ##


if __name__ == "__main__":
    main()
