import os
from collections import defaultdict
from typing import List, Dict

from dataset_specific.msmarco.common import load_query_group, top100_doc_ids, \
    QueryID
from epath import job_man_dir
from misc_lib import TimeEstimator


def main():
    split = "train"
    save_dir = os.path.join(job_man_dir, "MSMARCO_{}_title_body_tokens_working".format(split))
    num_corpus_job = 3213+1
    query_group: List[List[QueryID]] = load_query_group(split)
    candidate_docs: Dict[QueryID, List[str]] = top100_doc_ids(split)

    num_split_job = len(query_group) + 1
    print("Building doc_id_to_job_id_list")
    doc_id_to_job_id_list = defaultdict(list)
    for job_id, qids in enumerate(query_group):
        for qid in qids:
            try:
                docs = candidate_docs[qid]
                for doc_id in docs:
                    doc_id_to_job_id_list[doc_id].append(job_id)
            except KeyError as e:
                print(e)


    file_ptr_list = []
    for job_id in range(num_split_job):
        f = open(os.path.join(save_dir, str(job_id)), "w")
        file_ptr_list.append(f)

    ticker = TimeEstimator(num_corpus_job)
    for job_id in range(num_corpus_job):
        ticker.tick()
        token_path = os.path.join(job_man_dir, "MSMARCO_tokens", str(job_id))
        read_f = open(token_path, "r")
        print("Corpus job {}".format(job_id))
        for line in read_f:
            idx = line.index("\t")
            doc_id = line[:idx]
            print(doc_id)
            for job_id in doc_id_to_job_id_list[doc_id]:
                f = file_ptr_list[job_id]
                f.write(line)


    for f in file_ptr_list:
        f.close()

if __name__ == "__main__":
    main()