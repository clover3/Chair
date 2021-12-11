import os

from arg.counter_arg_retrieval.build_dataset.run3.run_interface.future_scorer_bert_like import FutureScorerBertLike
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.passage_scorer import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.run3.run_interface.run3_util import load_premise_queries, run_job_runner, \
    Run3PassageScoring
from arg.counter_arg_retrieval.build_dataset.run3.swtt.nli_common import EncoderForNLI
from bert_api.predictor import Predictor, PredictorWrap
from cpath import output_path


def main():
    model_path = os.path.join(output_path, "model", "runs", "standard_nli")
    max_seq_length = 512
    inner_predictor = Predictor(model_path, 3, max_seq_length)
    predictor = PredictorWrap(inner_predictor, lambda x: x[2])
    query_list = load_premise_queries()
    scorer: FutureScorerI = FutureScorerBertLike(predictor, EncoderForNLI, max_seq_length)
    scoring = Run3PassageScoring(scorer)
    run_job_runner(query_list, scoring.work, "PQ_3")


if __name__ == "__main__":
    main()
