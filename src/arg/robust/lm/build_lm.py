import os
import pickle
from collections import Counter
from typing import List, Iterable, Dict

from base_type import FilePath
from data_generator.data_parser.robust2 import load_robust_qrel
from epath import job_man_dir
from list_lib import lmap
from misc_lib import get_dir_files, tprint
from models.classic.lm_util import merge_lms


def load_counter_dict():
    d = {}
    dir_path = FilePath(os.path.join(job_man_dir, "robust_tokens"))
    for file_path in get_dir_files(dir_path):
        d.update(pickle.load(open(file_path, "rb")))
    return d


def main():
    tprint("loading counter dict")
    counter_dict: Dict[str, Counter] = load_counter_dict()

    def get_doc_lm(doc_id) -> Counter:
        counter = counter_dict[doc_id]
        n_tf = sum(counter.values())
        out_counter = Counter()
        for word, cnt in counter.items():
            out_counter[word] = cnt / n_tf
        return out_counter

    qrel = load_robust_qrel()

    def get_pos_docs(query_id):
        if query_id not in qrel:
            return
        judgement = qrel[query_id]
        for doc_id, score in judgement.items():
            if score:
                yield doc_id

    tprint("build query lm dict")
    query_lm_dict = {}
    queries = list(qrel.keys())
    for query_id in queries:
        pos_docs_ids: Iterable[str] = get_pos_docs(query_id)
        pos_doc_lms: List[Counter] = lmap(get_doc_lm, pos_docs_ids)
        query_lm: Counter = merge_lms(pos_doc_lms)
        query_lm_dict[query_id] = query_lm


if __name__ == "__main__":
    main()
