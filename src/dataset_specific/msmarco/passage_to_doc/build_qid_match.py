

# all_qids_in_doc_corpus
# qids_that_match
# passage_idx for qids
# docs for
from typing import List, Tuple

from dataset_specific.msmarco.common import load_queries, QueryID
from dataset_specific.msmarco.passage_to_doc.resource_loader import load_qrel
from list_lib import left
from misc_lib import tprint


def main():
    split = "train"
    passage_qrel = load_qrel(split)
    tprint("Passage qrel has {} entries".format(len(passage_qrel)))
    all_qids_in_doc_corpus = left(load_queries(split))
    tprint("There are {} doc queries".format(len(all_qids_in_doc_corpus)))
    tprint("joininig two qids")
    qids_that_match = list([qid for qid in all_qids_in_doc_corpus if qid in passage_qrel])
    tprint("There are {} common queries".format(len(qids_that_match)))
    PassageID = str
    positive_passage_list: List[Tuple[QueryID, PassageID]] = []

    #
    for qid in qids_that_match:
        for passage_id, score in passage_qrel[qid].items():
            if score:
                positive_passage_list.append((qid, passage_id))
    # print(len(positive_passage_list))
    # save_to_pickle(positive_passage_list, "msmarco_doc_joined_passage_list")


if __name__ == "__main__":
    main()