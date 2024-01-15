from typing import Tuple, Iterator

from adhoc.misc_helper import group_pos_neg, enumerate_pos_neg_pairs_once
from dataset_specific.msmarco.passage.passage_resource_loader import load_qrel
from dataset_specific.msmarco.passage.processed_resource_loader import get_queries_path
from list_lib import left
from misc_lib import path_join, group_iter
from table_lib import tsv_iter


def iter_dev_split_sample_pairwise(split_name="dev_sample1000_B") -> Iterator[Tuple[str, str, str]]:
    qrels_dict = load_qrel("dev")
    source_corpus_path = path_join("data", "msmarco", "top1000.dev")
    itr = group_iter(tsv_iter(source_corpus_path), lambda x: x[0])
    items = tsv_iter(get_queries_path(split_name))
    target_qids = set(left(items))

    for entries in itr:
        qid = entries[0][0]
        if qid not in target_qids:
            continue

        pos_doc_ids = []
        for doc_id, label in qrels_dict[qid].items():
            if label > 0:
                pos_doc_ids.append(doc_id)

        def is_pos(entry):
            qid, doc_id, _, _ = entry
            return doc_id in pos_doc_ids

        pos_items, neg_items = group_pos_neg(entries, is_pos)

        for pos, neg in enumerate_pos_neg_pairs_once((pos_items, neg_items)):
            qid, doc_id, query, pos_doc_text = pos
            qid_, doc_id_, query_, neg_doc_text = neg
            yield query, pos_doc_text, neg_doc_text
