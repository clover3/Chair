import os

from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.passage_scoring.future_scorer import FutureSWTTScorerBertLike
from arg.counter_arg_retrieval.build_dataset.run3.swtt.nli_common import EncoderForNLI
from bert_api.msmarco_tokenization import EncoderUnit
from bert_api.predictor import Predictor, PredictorWrap
from cpath import output_path


def get_msmarco_future_scorer() -> FutureScorerI:
    model_path = os.path.join(output_path, "model", "runs", "BERT_Base_trained_on_MSMARCO", )
    max_seq_length = 512
    inner_predictor = Predictor(model_path, 2, max_seq_length, False)
    predictor = PredictorWrap(inner_predictor, lambda x: x[1])
    scorer: FutureScorerI = FutureSWTTScorerBertLike(predictor, EncoderUnit, max_seq_length)
    return scorer


def get_nli_scorer():
    model_path = os.path.join(output_path, "model", "runs", "standard_nli")
    max_seq_length = 512
    inner_predictor = Predictor(model_path, 3, max_seq_length, False)
    predictor = PredictorWrap(inner_predictor, lambda x: x[2])
    scorer: FutureScorerI = FutureSWTTScorerBertLike(predictor, EncoderForNLI, max_seq_length)
    return scorer


