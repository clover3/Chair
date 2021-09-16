import json
import os
import random
import sys
from typing import List, Tuple

from cpath import output_path
from trec.trec_parse import scores_to_ranked_list_entries, write_trec_ranked_list_entry


def main():
    save_name = sys.argv[1]
    info_path = sys.argv[2]
    run_name = "random"
    info_d = json.load(open(info_path, "r", encoding="utf-8"))
    deletion_per_job = 20
    num_jobs = 10
    tag_type = "mismatch"
    save_path = os.path.join(output_path, "alamri_annotation1", "ranked_list", save_name + ".txt")

    max_offset = num_jobs * deletion_per_job
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
            doc_id_score_list: List[Tuple[str, float]]  = [(str(idx), random.random()) for idx, _ in enumerate(tokens)]
            query_id = "{}_{}_{}_{}".format(info['group_no'], info['inner_idx'], sent_name, tag_type)
            ranked_list = scores_to_ranked_list_entries(doc_id_score_list, run_name, query_id)
            all_ranked_list.extend(ranked_list)

    write_trec_ranked_list_entry(all_ranked_list, save_path)


if __name__ == "__main__":
    main()