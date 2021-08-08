import base64
import csv
import json
import os
from typing import List, Dict

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopicv2, CaTopic, get_ca2_converter
from arg.counter_arg_retrieval.build_dataset.resources import load_step2_claims_as_ca_topic
from arg.counter_arg_retrieval.build_dataset.run1.select_common_topk import get_candidate_docs
from cpath import output_path
from list_lib import lmap, right, foreach


def save_todo_list(all_entries, duplicate_doc_ids, topics_ca_id_index):
    output_rows = []
    for qid, entries in all_entries:
        topic: CaTopicv2 = topics_ca_id_index[qid]
        c_text = topic.claim_text
        target_p_texts: List[str] = right(topic.target_p)
        other_p_texts: List[List[str]] = lmap(right, topic.other_ps)
        j = {
            'target_claim': target_p_texts,
            'claims': other_p_texts
        }
        json_str = json.dumps(j)
        claim_b64_bytes = base64.b64encode(json_str.encode("utf-8"))
        base64_str = claim_b64_bytes.decode('utf-8')

        for doc_id in entries:
            if doc_id in duplicate_doc_ids:
                continue

            row = [c_text, base64_str, doc_id]
            output_rows.append(row)
    save_path = os.path.join(output_path, "ca_building", "run1", "todo_ca6.csv")
    csv_writer = csv.writer(open(save_path, "w", newline='', encoding="utf-8"))
    foreach(csv_writer.writerow, output_rows)


def save_to_readable_entries(all_entries, duplicate_doc_ids, topics_ca_id_index):
    doc_id_to_topic = []
    p_text_rows = []
    for qid, entries in all_entries:
        topic: CaTopicv2 = topics_ca_id_index[qid]
        c_text = topic.claim_text
        target_p_texts: List[str] = right(topic.target_p)
        other_p_texts: List[List[str]] = lmap(right, topic.other_ps)

        p_text_rows.append("Topic {}".format(c_text))
        for idx, p_texts in enumerate([target_p_texts] + other_p_texts):
            p_text_rows.append(str(idx))
            for p_text in p_texts:
                p_text_rows.append(p_text)

        for doc_id in entries:
            if doc_id in duplicate_doc_ids:
                continue

            row = [c_text, doc_id]
            doc_id_to_topic.append(row)
    p_text_rows = [[r] for r in p_text_rows]
    save_path_doc_topic = os.path.join(output_path, "ca_building", "run1", "todo_ca6_doc_to_topic.txt")
    save_path_p_text = os.path.join(output_path, "ca_building", "run1", "todo_ca6_p_text.txt")
    save_csv_to_path(save_path_doc_topic, doc_id_to_topic)
    save_csv_to_path(save_path_p_text, p_text_rows)


def save_csv_to_path(save_path, output_rows):
    csv_writer = csv.writer(open(save_path, "w", newline='', encoding="utf-8"))
    foreach(csv_writer.writerow, output_rows)


def main():
    all_entries, duplicate_doc_ids = get_candidate_docs()

    ca1_topics: List[CaTopic] = load_step2_claims_as_ca_topic()

    ca2_topics = lmap(get_ca2_converter(), ca1_topics)
    topics_ca_id_index: Dict[str, CaTopicv2] = {ca.ca_cid: ca for ca in ca2_topics}
    save_to_readable_entries(all_entries, duplicate_doc_ids, topics_ca_id_index)
    # save_todo_list(all_entries, duplicate_doc_ids, topics_ca_id_index)


if __name__ == "__main__":
    main()