import csv
import os
from typing import Dict, Tuple, List

from arg.counter_arg_retrieval.build_dataset.run3.run_interface.passage_scorer import rerank_passages, FutureScorerI
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.split_documents import PassageRange, \
    load_ca3_swtt_passage
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cpath import output_path
from data_generator.job_runner import ListWorker
from job_manager.job_runner_with_server import JobRunnerS
from trec.trec_parse import load_ranked_list_grouped
from trec.types import DocID


class Run3PassageScoring:
    def __init__(self, scorer: FutureScorerI):
        rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.filtered.txt")
        self.rlg = load_ranked_list_grouped(rlg_path)
        self.doc_as_passage_dict: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] \
            = load_ca3_swtt_passage()
        self.scorer: FutureScorerI = scorer

    def work(self, query_list):
        future_output = rerank_passages(self.doc_as_passage_dict, self.rlg, query_list, self.scorer)
        output = []
        for ca_query, docs_and_scores_future in future_output:
            docs_and_scores = [(doc_id, doc_scores_future.get())
                               for doc_id, doc_scores_future in docs_and_scores_future]
            output.append((ca_query, docs_and_scores))
        return output


def run_job_runner(ca_query_list, work_fn, job_name):
    def factory(out_dir):
        return ListWorker(work_fn, ca_query_list, out_dir)

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