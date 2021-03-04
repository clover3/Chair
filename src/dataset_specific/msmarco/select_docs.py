import os
import random
from typing import List, Iterable, Callable, Dict, Tuple, Set
from typing import Set

from dataset_specific.msmarco.common import per_query_root, get_per_query_doc_path, MSMarcoDataReader, open_top100, \
    root_dir
from misc_lib import get_second, exist_or_mkdir, tprint, TimeEstimator


def select_doc_per_query(split):
    ms_reader = MSMarcoDataReader(split)
    save_path = os.path.join(root_dir, "train_docs_10times_{}.tsv".format(split))
    out_f = open(save_path, "w")
    
    def pop(query_id, cur_doc_ids: Set):
        pos_docs = ms_reader.qrel[query_id]
        neg_docs = list([doc_id for doc_id in cur_doc_ids if doc_id not in pos_docs])
        if pos_docs:
            num_neg_docs = 10 * len(pos_docs)
            sel_docs = random.sample(neg_docs, num_neg_docs)
            doc_needed = pos_docs + sel_docs
            row = [query_id] + doc_needed
            out_f.write("\t".join(row) + "\n")

    total_line = 36701116
    ticker = TimeEstimator(total_line, "reading", 1000)
    with open_top100(split) as top100f:
        last_topic_id = None
        cur_doc_ids = set()
        for line_no, line in enumerate(top100f):
            [topic_id, _, doc_id, rank, _, _] = line.split()
            if last_topic_id is None:
                last_topic_id = topic_id
            elif last_topic_id != topic_id:
                pop(last_topic_id, cur_doc_ids)
                last_topic_id = topic_id
                cur_doc_ids = set()

            ticker.tick()
            cur_doc_ids.add(doc_id)


def select_doc_per_query_top50(split):
    ms_reader = MSMarcoDataReader(split)
    save_path = os.path.join(root_dir, "train_docs_top50_{}.tsv".format(split))
    out_f = open(save_path, "w")

    def pop(query_id, cur_doc_ids: List[Tuple[str, int]]):
        pos_docs = ms_reader.qrel[query_id]
        neg_docs = []
        for doc_id, rank in cur_doc_ids:
            if doc_id not in pos_docs and rank < 50:
                neg_docs.append(doc_id)
        doc_needed = pos_docs + neg_docs
        row = [query_id] + doc_needed
        out_f.write("\t".join(row) + "\n")

    total_line = 36701116
    ticker = TimeEstimator(total_line, "reading", 1000)
    with open_top100(split) as top100f:
        last_topic_id = None
        cur_doc_ids = []
        for line_no, line in enumerate(top100f):
            [topic_id, _, doc_id, rank, _, _] = line.split()
            if last_topic_id is None:
                last_topic_id = topic_id
            elif last_topic_id != topic_id:
                pop(last_topic_id, cur_doc_ids)
                last_topic_id = topic_id
                cur_doc_ids = []

            ticker.tick()
            cur_doc_ids.append((doc_id, int(rank)))

##
def main():
    select_doc_per_query_top50("train")

    # select_doc_per_query("dev")



if __name__ == "__main__":
    main()