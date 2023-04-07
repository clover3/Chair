import random

from dataset_specific.msmarco.passage.grouped_reader import get_train_query_grouped_dict_10K
from dataset_specific.msmarco.passage.passage_resource_loader import load_qrel
from typing import List, Iterable, Callable, Dict, Tuple, Set


def enum_pos_neg_sample(group_range: Iterable):
    qrel = load_qrel("train")

    def split_pos_neg_entries(qid, entries):
        pos_doc_ids = []
        for doc_id, score in qrel[qid].items():
            if score > 0:
                pos_doc_ids.append(doc_id)

        pos_doc = []
        neg_doc = []
        for e in entries:
            qid, pid, query, text = e
            if pid in pos_doc_ids:
                pos_doc.append(e)
            else:
                neg_doc.append(e)
        return pos_doc, neg_doc

    for group_no in group_range:
        d = get_train_query_grouped_dict_10K(group_no)
        for query_id, entries in d.items():
            try:
                pos_docs, neg_docs = split_pos_neg_entries(query_id, entries)
                for pos_entry in pos_docs:
                    neg_idx = random.randrange(len(neg_docs))
                    neg_entry = neg_docs[neg_idx]
                    query_text = pos_entry[2]
                    pos_text = pos_entry[3]
                    neg_text = neg_entry[3]
                    yield query_text, pos_text, neg_text
            except ValueError as e:
                print("Entries:", len(entries))
                print(e)


def check_ranked_list_size(group_range: Iterable):
    n_query = 0
    n_miss = 0
    for group_no in group_range:
        d = get_train_query_grouped_dict_10K(group_no)
        for query_id, entries in d.items():
            n_query += 1
            if len(entries) == 1:
                n_miss += 1
                print("{} of {} queries have 1 documents".format(n_miss, n_query))
