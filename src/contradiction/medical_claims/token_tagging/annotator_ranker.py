import json
import os
from typing import List, Dict, Tuple

from cpath import output_path
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import scores_to_ranked_list_entries, write_trec_ranked_list_entry
from trec.types import QRelsDict, DocID


def main():
    info_path = os.path.join(output_path, "alamri_annotation1", "tfrecord", "biobert_alamri1.info")
    qrel_path = os.path.join(output_path, "alamri_annotation1", "label", "worker_Q.qrel")
    # tag_type = "mismatch"
    tag_type = "conflict"
    save_name = "annotator_q_" + tag_type
    make_ranked_list(save_name, info_path, qrel_path, tag_type)


def make_ranked_list(save_name, info_path, source_qrel_path, tag_type):
    qrel: QRelsDict = load_qrels_structured(source_qrel_path)
    run_name = "annotator"
    info_d = json.load(open(info_path, "r", encoding="utf-8"))
    save_path = os.path.join(output_path, "alamri_annotation1", "ranked_list", save_name + ".txt")

    all_ranked_list = []
    for data_id, info in info_d.items():
        text1 = info['text1']
        text2 = info['text2']
        todo = [
            (text1, 'prem'),
            (text2, 'hypo'),
        ]
        for text, sent_name in todo:
            tokens = text.split()
            query_id = "{}_{}_{}_{}".format(info['group_no'], info['inner_idx'], sent_name, tag_type)
            if query_id not in qrel:
                print("Query {} not found".format(query_id))
                continue
            qrel_entries: Dict[DocID, int] = qrel[query_id]

            def get_score(doc_id)-> float:
                return 1 if doc_id in qrel_entries else 0
            doc_id_score_list: List[Tuple[str, float]]\
                = [(str(idx), get_score(str(idx))) for idx, _ in enumerate(tokens)]

            ranked_list = scores_to_ranked_list_entries(doc_id_score_list, run_name, query_id)
            all_ranked_list.extend(ranked_list)
    write_trec_ranked_list_entry(all_ranked_list, save_path)


if __name__ == "__main__":
    main()