import os
import pickle
from typing import List

from cpath import at_data_dir
from dataset_specific.msmarco.common import QueryID, load_query_group
from epath import job_man_dir
from misc_lib import exist_or_mkdir, TimeEstimator
from trec.qrel_parse import load_qrels_structured
from trec.types import QRelsDict


def group_passage_per_job(save_dir, query_group, passage_qrels):
    qid_to_job_id = {}
    pid_to_job_id = {}

    passage_per_job_id = {}
    for job_id, q_group in enumerate(query_group):
        passage_per_job_id[job_id] = {}
        for qid in q_group:
            for passage_id, score in passage_qrels[qid].items():
                pid_to_job_id[passage_id] = job_id
            qid_to_job_id[qid] = job_id

    msmarco_passage_corpus_path = at_data_dir("msmarco", "collection.tsv")
    ticker = TimeEstimator(80 * 1000 * 1000, "read", 1000 * 80)
    cnt = 0
    with open(msmarco_passage_corpus_path, 'r', encoding='utf8') as f:
        for line in f:
            passage_id, text = line.split("\t")
            if passage_id in pid_to_job_id:
                job_id = pid_to_job_id[passage_id]
                passage_per_job_id[job_id][passage_id] = text

            cnt += 1
            ticker.tick()

    for job_id, _ in enumerate(query_group):
        save_path = os.path.join(save_dir, str(job_id))
        pickle.dump(passage_per_job_id[job_id], open(save_path, "wb"))


if __name__ == "__main__":
    split = "train"
    msmarco_passage_qrel_path = at_data_dir("msmarco", "qrels.{}.tsv".format(split))
    passage_qrels: QRelsDict = load_qrels_structured(msmarco_passage_qrel_path)

    query_group: List[List[QueryID]] = load_query_group(split)
    save_dir = os.path.join(job_man_dir, "passage_join_for_{}".format(split))
    ##
    exist_or_mkdir(save_dir)

    group_passage_per_job(save_dir, query_group, passage_qrels)


    ###

