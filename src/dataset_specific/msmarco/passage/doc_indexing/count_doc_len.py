from cache import save_to_pickle
from dataset_specific.msmarco.passage.doc_indexing.resource_loader import enum_msmarco_passage_tokenized


def main():
    dl_d = {}
    for doc_id, terms in enum_msmarco_passage_tokenized():
        dl = len(terms)
        dl_d[doc_id] = dl

    save_to_pickle(dl_d, "msmarco_passage_dl_d")


if __name__ == "__main__":
    main()