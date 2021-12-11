import csv
import json
import os
from typing import Dict, Tuple, List

from arg.counter_arg_retrieval.build_dataset.run3.run_interface.passage_scorer import rerank_passages, FutureScorerI
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.split_documents import PassageRange, \
    load_ca3_swtt_passage
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from cpath import output_path
from data_generator.job_runner import ListWorker, WorkerInterface
from job_manager.job_runner_with_server import JobRunnerS
from list_lib import lmap
from trec.trec_parse import load_ranked_list_grouped
from trec.types import DocID


def scoring_output_to_json(output: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]):
    def doc_and_score_to_json(doc_n_score: Tuple[str, SWTTScorerOutput]):
        doc_id, score = doc_n_score
        return {
            'doc_id': doc_id,
            'score': score.to_json()
        }

    def qid_and_docs_to_json(qid_and_doc: Tuple[str, List[Tuple[str, SWTTScorerOutput]]]):
        qid, docs = qid_and_doc
        docs_json = list(map(doc_and_score_to_json, docs))
        return {
            'qid': qid,
            'docs': docs_json
        }
    return lmap(qid_and_docs_to_json, output)


def json_to_scoring_output(j) -> List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]:
    def parse_docs_and_scores(j_docs_and_scores) -> Tuple[str, SWTTScorerOutput]:
        return j_docs_and_scores['doc_id'], SWTTScorerOutput.from_json(j_docs_and_scores['score'])

    def parse_qid_and_docs(j_qid_and_docs) -> Tuple[str, List[Tuple[str, SWTTScorerOutput]]]:
        return j_qid_and_docs['qid'], lmap(parse_docs_and_scores, j_qid_and_docs['docs'])

    return lmap(parse_qid_and_docs, j)


class Run3PassageScoring:
    def __init__(self, scorer: FutureScorerI):
        rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.filtered.dummy.txt")
        self.rlg = load_ranked_list_grouped(rlg_path)
        self.doc_as_passage_dict: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] \
            = load_ca3_swtt_passage()
        self.scorer: FutureScorerI = scorer

    def work(self, query_list) -> List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]]:
        future_output = rerank_passages(self.doc_as_passage_dict, self.rlg, query_list, self.scorer)
        output: List[Tuple[str, List[Tuple[str, SWTTScorerOutput]]]] = []
        for ca_query, docs_and_scores_future in future_output:
            docs_and_scores = [(doc_id, doc_scores_future.get())
                               for doc_id, doc_scores_future in docs_and_scores_future]
            output.append((ca_query, docs_and_scores))
        return output


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