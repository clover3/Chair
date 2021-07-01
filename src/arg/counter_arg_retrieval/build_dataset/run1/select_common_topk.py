import csv
import os
from typing import List, Dict, Tuple

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic
from arg.counter_arg_retrieval.build_dataset.resources import load_step2_claims_as_ca_topic
from cpath import output_path
from list_lib import lmap, right, foreach
from misc_lib import get_duplicate_list
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def top_k_combine_by_and():
    ranking1 = os.path.join(output_path, "ca_building", "run1", "msmarco_ranked_list.txt")
    ranking2 = os.path.join(output_path, "ca_building", "q_res", "q_res_all")

    k = 50
    rlg1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranking1)
    rlg2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranking2)

    common_entries: List[Tuple[str, List[str]]] = []
    for q in rlg1:
        rl1 = rlg1[q]
        rl2 = rlg2[q]

        def get_top_k_doc_id(rl: List[TrecRankedListEntry]):
            return lmap(TrecRankedListEntry.get_doc_id, rl[:k])

        doc_ids1 = get_top_k_doc_id(rl1)
        doc_ids2 = get_top_k_doc_id(rl2)
        doc_ids_common = list([doc_id for doc_id in doc_ids1 if doc_id in doc_ids2])
        e = q, doc_ids_common
        print(q, len(doc_ids_common))
        common_entries.append(e)

    total_docs = sum(map(len, right(common_entries)))
    print("{} total docs from {} queries".format(total_docs, len(rlg1)))


def top_k_combine_by_or():
    ranking1 = os.path.join(output_path, "ca_building", "run1", "msmarco_ranked_list.txt")
    ranking2 = os.path.join(output_path, "ca_building", "q_res", "q_res_all")

    k = 5
    rlg1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranking1)
    rlg2: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranking2)

    all_entries: List[Tuple[str, List[str]]] = []
    for q in rlg1:
        rl1 = rlg1[q]
        rl2 = rlg2[q]

        def get_top_k_doc_id(rl: List[TrecRankedListEntry]):
            return lmap(TrecRankedListEntry.get_doc_id, rl[:k])

        doc_ids1 = get_top_k_doc_id(rl1)
        doc_ids2 = get_top_k_doc_id(rl2)
        doc_ids_all = set(doc_ids1)
        doc_ids_all.update(doc_ids2)
        e = q, list(doc_ids_all)
        all_entries.append(e)

    total_docs = sum(map(len, right(all_entries)))

    doc_ids_all = set()
    for docs in right(all_entries):
        doc_ids_all.update(docs)

    duplicate_doc_ids = get_duplicate_list(doc_ids_all)
    topics: List[CaTopic] = load_step2_claims_as_ca_topic()

    topics_ca_id_index: Dict[str, CaTopic] = {ca.ca_cid: ca for ca in topics}
    print("{} total docs from {} queries".format(total_docs, len(rlg1)))

    output_rows = []
    for qid, entries in all_entries:
        topic = topics_ca_id_index[qid]
        c_text = topic.claim_text
        p_text = topic.p_text
        for doc_id in entries:
            if doc_id in duplicate_doc_ids:
                continue

            row = [c_text, p_text, doc_id]
            output_rows.append(row)

    save_path = os.path.join(output_path, "ca_building", "run1", "mturk_todo.csv")
    csv_writer = csv.writer(open(save_path, "w", newline='', encoding="utf-8"))
    foreach(csv_writer.writerow, output_rows)



if __name__ == "__main__":
    top_k_combine_by_or()
