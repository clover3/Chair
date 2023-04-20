import random

from dataset_specific.msmarco.passage.grouped_reader import get_train_query_grouped_dict_10K
from dataset_specific.msmarco.passage.passage_resource_loader import MMPPosNegSampler
from typing import Iterable

from trainer_v2.chair_logging import c_log


def enum_pos_neg_sample(group_range: Iterable):
    pos_neg_sampler = MMPPosNegSampler()
    for group_no in group_range:
        try:
            c_log.info("Loading from group %s", group_no)
            d = get_train_query_grouped_dict_10K(group_no)
            c_log.info("Done")
            for query_id, entries in d.items():
                try:
                    pos_docs, neg_docs = pos_neg_sampler.split_pos_neg_entries(entries, query_id)
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
        except FileNotFoundError as e:
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
