from typing import Dict

from krovetzstemmer import Stemmer

from cache import load_from_pickle, save_to_pickle
from dataset_specific.msmarco.passage.doc_indexing.build_inverted_index_msmarco import InvIndex, \
    mmp_inv_index_ignore_voca
from trainer_v2.chair_logging import c_log


def apply_stem_to_inv_index_save():
    c_log.info("Loading pickle")
    inv_index = load_from_pickle("mmp_inv_index_lower")
    c_log.info("Applying stemming")
    new_inv_index = apply_stem_to_inv_index(inv_index)
    c_log.info("Saving pickle")
    save_to_pickle(new_inv_index, "mmp_inv_index_krovetz2")


def apply_stem_to_inv_index(inv_index) -> InvIndex:
    new_inv_index = {}
    ignore_voca = mmp_inv_index_ignore_voca()
    stemmer = Stemmer()
    for term, postings in inv_index.items():
        new_term = stemmer.stem(term)
        if term in ignore_voca or new_term in ignore_voca:
            continue

        if new_term in new_inv_index:
            # merge with existing entries
            prev_entries: Dict[str, int] = dict(new_inv_index[new_term])

            for doc_id, cnt in postings:
                if doc_id in prev_entries:
                    new_cnt = prev_entries[doc_id] + cnt
                else:
                    new_cnt = cnt
                prev_entries[doc_id] = new_cnt

            new_postings = [(doc_id, cnt) for doc_id, cnt in prev_entries.items()]
            new_inv_index[new_term] = new_postings
        else:
            new_inv_index[new_term] = postings
    return new_inv_index


def main():
    apply_stem_to_inv_index_save()


if __name__ == "__main__":
    main()
