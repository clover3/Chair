import pickle
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from cache import load_pickle_from
from cpath import output_path
from misc_lib import path_join

from misc_lib import get_first


class IntMappedInvIndex:
    def __init__(self, term_d, doc_id_d, new_inv_index):
        self.term_d: Dict[str, int] = term_d
        self.doc_id_d: Dict[str, int] = doc_id_d
        self.doc_id_inv: Dict[int, str] = {v:k for k, v in doc_id_d.items()}
        self.int_inv_index = new_inv_index

    def get_entries(self, term: str) -> List[Tuple[int, int]]:
        if term in self.term_d:
            term_id = self.term_d[term]
            return self.int_inv_index[term_id]
        else:
            return []

    def convert_doc_id_i_to_s(self, doc_id_i: int) -> str:
        return self.doc_id_inv[doc_id_i]


def convert_inv_index_to_int(inv_index: Dict[str, List[Tuple[str, int]]]):
    term_d: Dict[str, int] = {}
    doc_id_d: Dict[str, int] = {}

    new_inv_index = {}
    for term, entries in inv_index.items():
        if term in term_d:
            term_id = term_d[term]
        else:
            term_id = len(term_d)
            term_d[term] = term_id

        new_entries = []
        for doc_id_str, cnt in entries:
            if doc_id_str in doc_id_d:
                doc_id_i = doc_id_d[doc_id_str]
            else:
                doc_id_i = len(doc_id_d)
                doc_id_d[doc_id_str] = doc_id_i

            new_entries.append((doc_id_i, cnt))

        new_entries.sort(key=get_first)
        new_inv_index[term_id] = new_entries

    return IntMappedInvIndex(term_d, doc_id_d, new_inv_index)


def main():
    target_dir = sys.argv[1]

    inv_index_path = path_join(target_dir, "inv_index.pickle")
    int_inv_index_path = path_join(target_dir, "int_inv_index.pickle")
    doc_id_d_path = path_join(target_dir, "doc_id_d.pickle")
    term_d_path = path_join(target_dir, "term_d.pickle")

    inv_index = load_pickle_from(inv_index_path)
    int_mapped = convert_inv_index_to_int(inv_index)
    pickle.dump(int_mapped.doc_id_inv, open(doc_id_d_path, "wb"))
    pickle.dump(int_mapped.term_d, open(term_d_path, "wb"))
    pickle.dump(int_mapped.int_inv_index, open(int_inv_index_path, "wb"))


if __name__ == "__main__":
    main()
