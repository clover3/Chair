import csv
import json
import os
from typing import Dict, Tuple, List

from arg.counter_arg_retrieval.build_dataset.passage_scoring.passage_scoring import scoring_output_to_json, \
    PassageScoringInner
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.passage_scorer import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.split_documents import load_ca3_swtt_passage
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cpath import output_path
from data_generator.job_runner import ListWorker, WorkerInterface
from job_manager.job_runner_with_server import JobRunnerS
from trec.trec_parse import load_ranked_list_grouped
from trec.types import DocID


class Run3PassageScoring(PassageScoringInner):
    def __init__(self, scorer: FutureScorerI):
        rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.filtered.dummy.txt")
        rlg = load_ranked_list_grouped(rlg_path)
        doc_as_passage_dict: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] \
            = load_ca3_swtt_passage()
        super(Run3PassageScoring, self).__init__(scorer, rlg, doc_as_passage_dict)


class ListJson(WorkerInterface):
    def __init__(self, work_fn, todo, out_dir):
        self.work_fn = work_fn
        self.todo = todo
        self.out_dir = out_dir

    def work(self, job_id):
        save_path = os.path.join(self.out_dir, "{}.json".format(str(job_id)))
        output = self.work_fn(self.todo[job_id:job_id+1])
        j = scoring_output_to_json(output)
        json.dump(j, open(save_path, "w"), indent=True)


def run_job_runner(ca_query_list, work_fn, job_name):
    def factory(out_dir):
        return ListWorker(work_fn, ca_query_list, out_dir)

    root_dir = os.path.join(output_path, "ca_building", "run3")
    job_runner = JobRunnerS(root_dir, len(ca_query_list), job_name, factory)
    job_runner.auto_runner()


def run_job_runner_json(ca_query_list, work_fn, job_name):
    def factory(out_dir):
        return ListJson(work_fn, ca_query_list, out_dir)

    root_dir = os.path.join(output_path, "ca_building", "run3")
    job_runner = JobRunnerS(root_dir, len(ca_query_list), job_name, factory)
    job_runner.auto_runner()


def load_tsv_query(file_path):
    data = []
    for row in csv.reader(open(file_path, "r"), delimiter="\t"):
        data.append((row[0], row[1]))
    return data


def load_premise_queries():
    query_path = os.path.join(output_path, "ca_building", "run3", "queries", "premise_query.tsv")
    return load_tsv_query(query_path)


def load_manual_queries():
    query_path = os.path.join(output_path, "ca_building", "run3", "queries", "manual_query.tsv")
    return load_tsv_query(query_path)