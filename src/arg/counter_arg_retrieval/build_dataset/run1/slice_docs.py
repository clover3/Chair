import json
import os
import sys
from json import JSONDecodeError
from typing import List

from cpath import at_output_dir, output_path
from misc_lib import TimeEstimator
from trec.trec_parse import load_ranked_list
from trec.types import TrecRankedListEntry


def slice_docs_and_save(doc_ids, jsonl_path, out_path):
    f_out = open(out_path, "w")

    ticker = TimeEstimator(len(doc_ids))
    for idx, line in enumerate(open(jsonl_path, "r")):
            ###
        if line.strip():
            try:
                j = json.loads(line, strict=False)
                if j['id'] in doc_ids:
                    f_out.write(line)
                    ticker.tick()
            except JSONDecodeError:
                print(line)
                raise


def get_qids():
    qids = []
    j_obj = json.load(open(at_output_dir("ca_building", "claims.step2.txt"), "r"))
    for topic in j_obj:
        qid = str(topic['ca_cid'])
        qids.append(qid)
    return qids


def get_doc_ids_for_qids(q_res_path, qids):
    doc_ids = set()
    rl: List[TrecRankedListEntry] = load_ranked_list(q_res_path)
    for e in rl:
        if e.query_id in qids:
            doc_ids.add(e.doc_id)

    return doc_ids


def main():
    jsonl_path = sys.argv[1]
    qids = get_qids()
    print(qids)
    q_res_path = os.path.join(output_path, "ca_building", "q_res", "q_res_all")
    doc_ids = get_doc_ids_for_qids(q_res_path, qids)
    print("{} doc ids".format(len(doc_ids)))
    ###
    out_path = sys.argv[2]
    slice_docs_and_save(doc_ids, jsonl_path, out_path)




if __name__ == "__main__":
    main()