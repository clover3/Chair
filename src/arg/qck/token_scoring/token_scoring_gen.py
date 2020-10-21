from collections import OrderedDict
from typing import List, Iterable, Callable, Tuple
from typing import NamedTuple

from arg.qck.decl import QCKQuery, KDP, QKUnit
from arg.qck.qck_worker import InstanceGenerator
from arg.qck.token_scoring.decl import ScoreVector
from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import flatten, lmap
from misc_lib import DataIDManager
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.bert_data_gen import create_float_feature


class TokenScoringInstance(NamedTuple):
    query_text: str
    doc_tokens: List[str]
    data_id: int
    score: ScoreVector


def score_vector_to_feature(score_vector: ScoreVector):
    depth = len(score_vector[0])
    flat_size = depth * len(score_vector)
    flat_vector = list(flatten(score_vector))
    assert len(flat_vector) == flat_size
    return create_float_feature(flat_vector)


def pad_score_vector(score_vector: ScoreVector, max_seq_length, seg1_length) -> ScoreVector:
    depth = len(score_vector[0])
    zero_elem = [[0.] * depth]

    score_vector: ScoreVector = zero_elem * (seg1_length + 2) + score_vector + zero_elem
    score_vector += (max_seq_length - len(score_vector)) * zero_elem
    return score_vector


class TokenScoringGen(InstanceGenerator):
    def __init__(self, get_score: Callable[[QCKQuery, KDP], ScoreVector]):
        self.max_seq_length = 512
        self.tokenizer = get_tokenizer()
        self.get_score: Callable[[QCKQuery, KDP], ScoreVector] = get_score

    def generate(self,
                 kc_candidate: Iterable[QKUnit],
                 data_id_manager: DataIDManager,
                 ) -> Iterable[TokenScoringInstance]:

        def convert(pair: Tuple[QCKQuery, List[KDP]]) -> Iterable[TokenScoringInstance]:
            query, passages = pair
            for passage in passages:
                info = {
                            'query': query,
                            'kdp': passage
                        }
                yield TokenScoringInstance(query.text, passage.tokens,
                                           data_id_manager.assign(info),
                                           self.get_score(query, passage)
                                           )

        return flatten(lmap(convert, kc_candidate))

    def tokenize_from_tokens_w_scores(self, tokens: List[str], scores: ScoreVector) -> Tuple[List[str], ScoreVector]:
        sub_tokens = []
        sub_token_scores: ScoreVector = []
        for t, score in zip(tokens, scores):
            score: List[float] = score
            ts = self.tokenizer.tokenize(t)
            sub_tokens.extend(ts)
            sub_token_scores.extend([score] * len(ts))
        return sub_tokens, sub_token_scores

    def encode_fn(self, inst: TokenScoringInstance) -> OrderedDict:
        max_seq_length = self.max_seq_length
        tokens1: List[str] = self.tokenizer.tokenize(inst.query_text)
        max_seg2_len = self.max_seq_length - 3 - len(tokens1)

        tokens2, scores = self.tokenize_from_tokens_w_scores(inst.doc_tokens, inst.score)
        tokens2 = tokens2[:max_seg2_len]
        scores: ScoreVector = scores[:max_seg2_len]
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]

        segment_ids = [0] * (len(tokens1) + 2) \
                      + [1] * (len(tokens2) + 1)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        features = get_basic_input_feature(self.tokenizer, max_seq_length, tokens, segment_ids)

        score_vector = pad_score_vector(scores, max_seq_length, len(tokens1))
        if len(score_vector) != max_seq_length:
            print(score_vector)
            print(len(score_vector))
            print(max_seq_length)
            print(len(scores))
            print(scores)
        assert len(score_vector) == max_seq_length
        features['label_ids'] = score_vector_to_feature(score_vector)
        features['data_id'] = create_int_feature([inst.data_id])
        return features
