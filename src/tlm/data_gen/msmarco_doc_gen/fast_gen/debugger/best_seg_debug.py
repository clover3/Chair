import json
import os
from typing import List, Dict

from cache import load_pickle_from
from epath import job_man_dir
from tlm.data_gen.msmarco_doc_gen.fast_gen.seg_resource import SRPerQuery
from tlm.estimator_output_reader import join_prediction_with_info


def main():
    info_dir = os.path.join(job_man_dir, "best_seg_prediction_gen_train_info")
    job_id = 0
    info_file_path = os.path.join(info_dir, str(job_id) + ".info")
    print(info_file_path)
    info = json.load(open(info_file_path, "r"))
    prediction_dir = "output/mmd_ss/mmd_Z_50000"
    prediction_file = os.path.join(prediction_dir, str(job_id) + ".score")
    pred_data: List[Dict] = join_prediction_with_info(prediction_file, info)

    target_qdid = ("1000633", "D144400")
    saved_entries = []
    for key, entry in info.items():
        if entry['qid'] == "1000633" and entry['doc_id'] == 'D144400':
            saved_entries.append(entry)
            print(entry)

    print('--')
    for entry in pred_data:
        if entry['qid'] == "1000633" and entry['doc_id'] == 'D144400':
            print(entry)

    qid = "1000633"
    sr_path = os.path.join(job_man_dir, "seg_resource_train", qid)
    sr_per_query: SRPerQuery = load_pickle_from(sr_path)

    for sr_per_query_doc in sr_per_query.sr_per_query_doc:
        if sr_per_query_doc.doc_id == "D144400":
            print("doc {} has {} segs".format(sr_per_query_doc.doc_id, len(sr_per_query_doc.segs)))



if __name__ == "__main__":
    main()