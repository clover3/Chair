import os

from arg.counter_arg_retrieval.build_dataset.job_running import run_job_runner
from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.passage_scoring.offline_scorer_bert_like import FutureScorerBertLike
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.run3_util import load_manual_queries
from arg.counter_arg_retrieval.build_dataset.run4.run4_util import Run4PassageScoring
from bert_api.msmarco_tokenization import EncoderUnit
from bert_api.predictor import Predictor, PredictorWrap
from cpath import output_path


def main():
    model_path = os.path.join(output_path, "model", "runs", "BERT_Base_trained_on_MSMARCO",)
    max_seq_length = 512
    inner_predictor = Predictor(model_path, 2, max_seq_length)
    predictor = PredictorWrap(inner_predictor, lambda x: x[1])
    query_list = load_manual_queries()
    scorer: FutureScorerI = FutureScorerBertLike(predictor, EncoderUnit, max_seq_length)
    scoring = Run4PassageScoring(scorer)
    run_job_runner(query_list, scoring.work, "PQ_7")


if __name__ == "__main__":
    main()
