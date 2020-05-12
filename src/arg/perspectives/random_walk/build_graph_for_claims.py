import sys
from typing import List

from arg.perspectives.clueweb_db import preload_docs, load_doc
from arg.perspectives.random_walk.count_coocurrence_graph import build_co_occurrence
from cache import save_to_pickle
from galagos.parse import load_galago_ranked_list
from list_lib import lmap
from misc_lib import TimeEstimator
from tlm.retrieve_lm.stem import CacheStemmer


def get_tokens_form_doc_ids(doc_ids):
    preload_docs(doc_ids)
    list_tokens: List[List[str]] = lmap(load_doc, doc_ids)
    return list_tokens


def work(q_res_path, save_name):
    ranked_list_d = load_galago_ranked_list(q_res_path)
    window_size = 10
    stemmer = CacheStemmer()
    print(q_res_path)

    ticker = TimeEstimator(len(ranked_list_d))
    r = []
    for claim_id, ranked_list in ranked_list_d.items():
        ticker.tick()
        doc_ids = list([e.doc_id for e in ranked_list])
        print("1")
        counter = build_co_occurrence(get_tokens_form_doc_ids(doc_ids), window_size, stemmer)
        print("2")
        r.append((claim_id, counter))

    save_to_pickle(r, save_name)


if __name__ == "__main__":
    work(sys.argv[1], sys.argv[2])

