import os

from arg.counter_arg_retrieval.build_dataset.job_running import run_job_runner
from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.passage_scoring.offline_scorer_bert_like import FutureScorerBertLike
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.run3_util import load_premise_queries
from arg.counter_arg_retrieval.build_dataset.run3.swtt.nli_common import EncoderForNLI
from arg.counter_arg_retrieval.build_dataset.run4.run4_util import Run4PassageScoring
from bert_api.predictor import Predictor, PredictorWrap
from cpath import output_path


def main():
    model_path = os.path.join(output_path, "model", "runs", "standard_nli")
    max_seq_length = 512
    inner_predictor = Predictor(model_path, 3, max_seq_length)
    predictor = PredictorWrap(inner_predictor, lambda x: x[2])
    query_list = load_premise_queries()
    scorer: FutureScorerI = FutureScorerBertLike(predictor, EncoderForNLI, max_seq_length)
    scoring = Run4PassageScoring(scorer)
    run_job_runner(query_list, scoring.work, "PQ_8")


if __name__ == "__main__":
    main()
