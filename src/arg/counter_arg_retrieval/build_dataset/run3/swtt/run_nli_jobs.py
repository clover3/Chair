import os
import pickle
import sys
from typing import List, Tuple, Iterator

from arg.counter_arg_retrieval.build_dataset.ca_query import load_ca_query_from_tsv, CAQuery
from arg.counter_arg_retrieval.build_dataset.run3.swtt.nli_common import EncoderForNLI
from arg.counter_arg_retrieval.build_dataset.run3.swtt.run_classifier import run_classifier, \
    ScoredDocument
from bert_api.predictor import Predictor, FloatPredictor, PredictorWrap
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.swtt_scorer import DocumentScorerSWTT
from bert_api.swtt.window_enum_policy import WindowEnumPolicyMinPop
from cache import load_from_pickle
from cpath import output_path
from data_generator.job_runner import WorkerInterface
from job_manager.job_runner_with_server import JobRunnerS


class Worker(WorkerInterface):
    def __init__(self, model_path, ca_query_list, out_dir):
        self.docs: List[Tuple[str, SegmentwiseTokenizedText]] = load_from_pickle("ca_run3_swtt")
        self.enum_policy = WindowEnumPolicyMinPop(30, 50)
        inner_predictor = Predictor(model_path, 3, 300)
        self.predictor: FloatPredictor = PredictorWrap(inner_predictor, lambda x: x[2])
        self.ca_query_list = ca_query_list
        self.out_dir = out_dir

    def work(self, job_id):
        save_path = os.path.join(self.out_dir, str(job_id))
        ca_query_list = self.ca_query_list[job_id:job_id+1]
        run_batch_predictions(ca_query_list, self.docs, self.predictor,
                              self.enum_policy, save_path)


def run_batch_predictions(ca_query_list, docs, predictor: FloatPredictor, enum_policy, save_path):
    document_scorer = DocumentScorerSWTT(predictor, EncoderForNLI, 300)
    output: Iterator[Tuple[CAQuery, List[ScoredDocument]]] \
        = run_classifier(docs, document_scorer, enum_policy, ca_query_list)
    save_data: List[Tuple[CAQuery, List[ScoredDocument]]] = []
    for ca_query, docs_and_scores_future in output:
        docs_and_scores = [(doc_id, doc_scores_future.get()) for doc_id, doc_scores_future in docs_and_scores_future]
        save_data.append((ca_query, docs_and_scores))

    pickle.dump(save_data, open(save_path, "wb"))


def main():
    model_path = sys.argv[1]
    tsv_path = sys.argv[2].strip()
    job_name = sys.argv[3]
    ca_query_list: List[CAQuery] = load_ca_query_from_tsv(tsv_path)

    def factory(out_dir):
        return Worker(model_path, ca_query_list, out_dir)

    root_dir = os.path.join(output_path, "ca_building", "run3")
    job_runner = JobRunnerS(root_dir, len(ca_query_list), job_name, factory)
    job_runner.auto_runner()


if __name__ == "__main__":
    main()
