from collections import defaultdict
from typing import List

from cache import load_pickle_from
from galagos.parse import write_qrels
from mturk.parse_util import HitResult


def load_hit_results():
    pickle_path = "C:\\work\\Code\\Chair\\output\\ca_building\\run3\\batch_result\\Batch_341257_batch_results.csv.patched.pickle"
    # Show statistics
    hits: List[HitResult] = load_pickle_from(pickle_path)
    return hits


def hit_to_sig(hit: HitResult):
    qid = hit.inputs['qid']
    new_doc_id = "{}_{}".format(hit.inputs['doc_id'], hit.inputs['passage_idx'])
    return qid, new_doc_id


def hit_to_label(hit: HitResult):
    return hit.outputs["Q13.on"]


def main():
    hits = load_hit_results()

    qid_grouped = defaultdict(list)
    for h in hits:
        qid, doc_id = hit_to_sig(h)
        label = hit_to_label(h)
        qid_grouped[qid].append((doc_id, label))

    save_path = "C:\\work\\Code\\Chair\\output\\ca_building\\run3\\qrel.txt"
    write_qrels(qid_grouped, save_path)


if __name__ == "__main__":
    main()
