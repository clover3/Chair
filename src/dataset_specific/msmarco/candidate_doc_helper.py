from typing import List, Dict

from cache import save_to_pickle
from dataset_specific.msmarco.common import QueryID, top100_doc_ids


def main():
    split = "train"
    print("Reading...")
    candidate_docs_d: Dict[QueryID, List[str]] = top100_doc_ids(split)
    print("Saving...")
    save_to_pickle(candidate_docs_d, "MMD_candidate_docs_d_{}".format(split))


if __name__ == "__main__":
    main()