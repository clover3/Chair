import sys
from typing import List, Tuple

from arg.counter_arg_retrieval.build_dataset.ca_query import load_ca_query_from_tsv, CAQuery
from arg.counter_arg_retrieval.build_dataset.run3.swtt.run_classifier import run_batch_predictions
from bert_api.predictor import Predictor, FloatPredictor, PredictorWrap
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.window_enum_policy import WindowEnumPolicyMinPop
from cache import load_from_pickle
from data_generator.job_runner import WorkerInterface
from job_manager.job_runner_with_server import JobRunnerS


class Worker(WorkerInterface):
    def __init__(self, model_path, tsv_path_prefix, out_dir):
        self.docs: List[Tuple[str, SegmentwiseTokenizedText]] = load_from_pickle("ca_run3_swtt")
        self.enum_policy = WindowEnumPolicyMinPop(30, 50)
        self.predictor: FloatPredictor = PredictorWrap(Predictor(model_path, 2), lambda x: x[1])
        self.tsv_path_prefix = tsv_path_prefix

    def work(self, job_id):
        tsv_path = self.tsv_path_prefix + "{0:02d}".format(job_id)
        ca_query_list: List[CAQuery] = load_ca_query_from_tsv(tsv_path)
        save_path = tsv_path + ".result"
        run_batch_predictions(ca_query_list, self.docs, self.predictor,
                              self.enum_policy, save_path)



def main():
    model_path = sys.argv[1]
    tsv_path_prefix = sys.argv[2].strip()
    n_jobs = int(sys.argv[3])
    job_name = sys.argv[4]

    def factory(out_dir):
        return Worker(model_path, tsv_path_prefix, out_dir)

    job_runner = JobRunnerS("/tmp", n_jobs, job_name, factory)
    job_runner.auto_runner()


if __name__ == "__main__":
    main()
