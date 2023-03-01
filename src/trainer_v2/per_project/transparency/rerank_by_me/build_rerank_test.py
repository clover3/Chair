import json
import random
from cpath import output_path, data_path
from misc_lib import path_join

# Load TREC 19 DL QRel
#   Sample 10 passages with rel: non-rel = 1:1
#   5 Queries

def main():
    qrel_path = path_join(data_path, "splade", "msmarco", "TREC_DL_2019", "qrel.json")
    save_path = path_join(output_path, "transparency", "msmarco", "rerank_candiate_ids.json")
    qrel = json.load(open(qrel_path, "r"))
    n = 5

    data = []
    for qid in qrel:
        doc_rels = qrel[qid]
        pos_ids = []
        neg_ids = []
        for doc_id, rel in doc_rels.items():
            if rel > 0:
                pos_ids.append(doc_id)
            else:
                neg_ids.append(doc_id)

        random.shuffle(pos_ids)
        random.shuffle(neg_ids)
        if len(pos_ids) < n or len(neg_ids) < n:
            continue
        data.append((qid, pos_ids[:n], neg_ids[:n]))

        if len(data) >= 5:
            break

    json.dump(data, open(save_path, "w"))


if __name__ == "__main__":
    main()