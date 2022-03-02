import scipy.special

from arg.counter_arg_retrieval.build_dataset.passage_scorer_common import FutureScorerI
from arg.counter_arg_retrieval.build_dataset.passage_scoring.future_scorer_swtt_bert_like import \
    OfflineSWTTScorerBertLike
from arg.counter_arg_retrieval.build_dataset.run3.swtt.nli_common import EncoderForNLI


def get_offline_nli_scorer():
    max_seq_length = 512

    def logit_to_score(logit):
        return scipy.special.softmax(logit)[2]
    scorer: FutureScorerI = OfflineSWTTScorerBertLike("nli", EncoderForNLI, max_seq_length, logit_to_score)
    return scorer


