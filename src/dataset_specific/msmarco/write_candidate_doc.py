import os
import pickle
from typing import List, Dict

from dataset_specific.msmarco.common import load_query_group, top100_doc_ids, QueryID
from epath import job_man_dir


def main(split):
    query_group = load_query_group(split)
    candidate_docs: Dict[QueryID, List[str]] = top100_doc_ids(split)
    num_jobs = len(query_group)
    for job_id in range(176, num_jobs):
        qids = query_group[job_id]
        save_path = os.path.join(job_man_dir, "MMD_{}_candidate_doc".format(split), str(job_id))
        candidate_docs_for_job = {}
        for qid in qids:
            try:
                candidate_docs_for_job[qid] = candidate_docs[qid]
            except KeyError:
                print("Query {} not found in ranked list".format(qid))
        pickle.dump(candidate_docs_for_job, open(save_path, "wb"))


if __name__ == "__main__":
    main("train")