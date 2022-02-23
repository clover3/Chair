import abc
from collections import defaultdict
from typing import NamedTuple, List, Iterator, Dict, Tuple, Callable, Any, Optional

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import index_by_fn
from tlm.data_gen.doc_encode_common import split_window_get_length
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo
from tlm.qtype.partial_relevance.segmented_text import SegmentedText, seg_to_text

ContributionSummaryDict = Dict[str, List[float]]


class SegmentedInstance(NamedTuple):
    text1: SegmentedText
    text2: SegmentedText

    def enum_seg_indice_pairs(self):
        for seg1_idx in range(self.text1.get_seg_len()):
            for seg2_idx in range(self.text2.get_seg_len()):
                yield seg1_idx, seg2_idx

    def get_drop_mask(self, seg1_idx, seg2_idx) -> np.array:
        drop_mask_per_q_seg = self.text2.get_empty_seg_mask()
        drop_mask_per_q_seg[seg2_idx] = 1
        drop_mask = [self.text2.get_empty_seg_mask() for _ in range(self.text1.get_seg_len())]
        drop_mask[seg1_idx] = drop_mask_per_q_seg
        drop_mask = np.stack(drop_mask)
        return drop_mask

    def get_empty_mask(self) -> np.ndarray:
        return np.stack([self.text2.get_empty_seg_mask() for _ in range(self.text1.get_seg_len())])

    def enum_token_idx_from_seg1_idx(self, seg_idx) -> Iterator[int]:
        yield from self.text1.seg_token_indices[seg_idx]

    def enum_token_idx_from_seg2_idx(self, seg_idx) -> Iterator[int]:
        yield from self.text2.seg_token_indices[seg_idx]

    def translate_mask(self, drop_mask: np.array) -> Dict[Tuple[int, int], int]:
        new_mask = {}
        for q_seg_idx in range(self.text1.get_seg_len()):
            for d_seg_idx in range(self.text2.get_seg_len()):
                v = drop_mask[q_seg_idx, d_seg_idx]
                if v:
                    for q_token_idx in self.text1.enum_token_idx_from_seg_idx(q_seg_idx):
                        for d_token_idx in self.enum_token_idx_from_seg2_idx(d_seg_idx):
                            k = q_token_idx, d_token_idx
                            new_mask[k] = int(v)
        return new_mask

    def translate_mask_d(self, drop_mask) -> Dict[Tuple[int, int], int]:
        new_mask = {}
        for q_seg_idx in range(self.text1.get_seg_len()):
            for d_seg_idx in range(self.text2.get_seg_len()):
                try:
                    v = drop_mask[q_seg_idx, d_seg_idx]
                    if v:
                        for q_token_idx in self.text1.enum_token_idx_from_seg_idx(q_seg_idx):
                            for d_token_idx in self.enum_token_idx_from_seg2_idx(d_seg_idx):
                                k = q_token_idx, d_token_idx
                                new_mask[k] = int(v)
                except KeyError:
                    pass
        return new_mask
    def accumulate_over(self, raw_scores, accumulate_method: Callable[[List[float]], float]):
        scores_d = defaultdict(list)
        for q_seg_idx in range(self.text1.get_seg_len()):
            for d_seg_idx in range(self.text2.get_seg_len()):
                key = q_seg_idx, d_seg_idx
                for q_token_idx in self.text1.enum_token_idx_from_seg_idx(q_seg_idx):
                    for d_token_idx in self.text2.enum_token_idx_from_seg_idx(d_seg_idx):
                        v = raw_scores[q_token_idx, d_token_idx]
                        scores_d[key].append(v)

        out_d = {}
        for q_seg_idx in range(self.text1.get_seg_len()):
            for d_seg_idx in range(self.text2.get_seg_len()):
                key = q_seg_idx, d_seg_idx
                scores = scores_d[key]
                out_d[key] = accumulate_method(scores)
        return out_d

    def score_d_to_table(self, contrib_score_d: Dict[Tuple[int, int], Any]):
        return self._score_d_to_table(contrib_score_d)

    def _score_d_to_table(self, contrib_score_table):
        table = []
        for q_seg_idx in range(self.text1.get_seg_len()):
            row = []
            for d_seg_idx in range(self.text2.get_seg_len()):
                key = q_seg_idx, d_seg_idx
                row.append(contrib_score_table[key])
            table.append(row)
        return table

    def score_np_table_to_table(self, contrib_score_table):
        return self._score_d_to_table(contrib_score_table)

    def get_seg2_dropped_instances(self, drop_indices):
        return SegmentedInstance(SegmentedText(self.text1.tokens_ids, self.text1.seg_token_indices),
                                 self.text2.get_dropped_text(drop_indices),
                                 )
    @classmethod
    def from_flat_args(cls,
                       text1_tokens_ids,
                       text2_tokens_ids,
                       text1_seg_indices,
                       text2_seg_indices,
                       ):
        return SegmentedInstance(SegmentedText(text1_tokens_ids, text1_seg_indices),
                                 SegmentedText(text2_tokens_ids, text2_seg_indices),
                                 )

    def to_json(self):
        return {
            'text1': self.text1.to_json(),
            'text2': self.text2.to_json(),
        }

    @classmethod
    def from_json(cls, j):
        return SegmentedInstance(SegmentedText.from_json(j['text1']),
                                 SegmentedText.from_json(j['text2']),
                                 )

    def str_hash(self) -> str:
        return str(self.to_json())


class RelatedEvalInstance(NamedTuple):
    problem_id: str
    query_info: QueryInfo
    seg_instance: SegmentedInstance
    score: float

    def to_json(self):
        return {
            'problem_id': self.problem_id,
            'query_info': self.query_info.to_json(),
            'seg_instance': self.seg_instance.to_json(),
            'score': self.score
        }

    @classmethod
    def from_json(cls, j):
        return RelatedEvalInstance(j['problem_id'],
                                   QueryInfo.from_json(j['query_info']),
                                   SegmentedInstance.from_json(j['seg_instance']),
                                   j['score']
                                   )


class RelatedEvalInstanceEx(NamedTuple):
    problem_id: str
    target_seg_idx: int
    seg_instance: SegmentedInstance
    score: float

    def to_json(self):
        return {
            'problem_id': self.problem_id,
            'target_seg_idx': self.target_seg_idx,
            'seg_instance': self.seg_instance.to_json(),
            'score': self.score
        }

    @classmethod
    def from_json(cls, j):
        return RelatedEvalInstanceEx(j['problem_id'],
                                   j['target_seg_idx'],
                                   SegmentedInstance.from_json(j['seg_instance']),
                                   j['score']
                                   )

def rei_to_text(tokenizer, rei: RelatedEvalInstance):
    seg1_text = seg_to_text(tokenizer, rei.seg_instance.text1)
    seg2_text = seg_to_text(tokenizer, rei.seg_instance.text2)
    return f"RelatedEvalInstance({rei.problem_id}, {seg1_text})\n" \
           + "Doc: " + seg2_text


class ContributionSummary(NamedTuple):
    table: List[List[float]]

    @classmethod
    def from_single_array(cls, arr: List[float], target_seg_idx, n_seg):
        output = []
        for i in range(n_seg):
            if i == target_seg_idx:
                output.append(arr)
            else:
                output.append([])
        return ContributionSummary(output)

    @classmethod
    def from_indices(cls, indices: List[int], target_seg_idx, p: RelatedEvalInstance):
        n = p.seg_instance.text2.get_seg_len()
        zeros = [0 for _ in range(n)]
        for i in indices:
            zeros[i] = 1

        return cls.from_single_array(zeros, target_seg_idx, p.seg_instance.text1.get_seg_len())


class RelatedEvalAnswer(NamedTuple):
    problem_id: str
    contribution: ContributionSummary

    @classmethod
    def from_indices(cls, indices: List[int], target_seg_idx, p: RelatedEvalInstance):
        return RelatedEvalAnswer(p.problem_id,
                                 ContributionSummary.from_indices(indices, target_seg_idx, p))


class RelatedBinaryAnswer(NamedTuple):
    problem_id: str
    score_table: List[List[int]]


class PerProblemEvalResult(NamedTuple):
    problem_id: str
    scores: List[Optional[float]]

    def to_json(self):
        return {
            'problem_id': self.problem_id,
            'scores': self.scores
        }

    @classmethod
    def from_json(cls, j):
        return PerProblemEvalResult(j['problem_id'], j['scores'])


def join_p_withother(problems: List[RelatedEvalInstance],
                     obj_list: List) -> List[Tuple[RelatedEvalInstance, Any]]:
    pid_to_obj = index_by_fn(lambda e: e.problem_id, obj_list)
    output = []
    for p in problems:
        c = pid_to_obj[p.problem_id]
        output.append((p, c))
    return output


def get_coffee_doc():
    doc = "The one of earliest credible evidence of the drinking of coffee in the form of the modern beverage appears in modern-day Yemen from the middle of the 15th century in Sufi shrines, where coffee seeds were first roasted and brewed in a manner similar to current methods.[2] The Yemenis procured the coffee beans from the Ethiopian Highlands via coastal Somali intermediaries and began cultivation. By the 16th century, the drink had reached the rest of the Middle East and North Africa, later spreading to Europe"
    return doc


def get_test_segment_instance() -> SegmentedInstance:
    query = 'Where is coffee from?'
    doc_short = "The one of earliest credible evidence of the drinking of coffee in the form of the modern beverage appears in modern-day Yemen from the middle of the 15th century in Sufi shrines,"
    doc = "The one of earliest credible evidence of the drinking of coffee in the form of the modern beverage appears in modern-day Yemen from the middle of the 15th century in Sufi shrines, where coffee seeds were first roasted and brewed in a manner similar to current methods.[2] The Yemenis procured the coffee beans from the Ethiopian Highlands via coastal Somali intermediaries and began cultivation. By the 16th century, the drink had reached the rest of the Middle East and North Africa, later spreading to Europe"
    tokenizer = get_tokenizer()
    q_tokens_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query))
    d_tokens = tokenizer.tokenize(doc)
    d_tokens_ids = tokenizer.convert_tokens_to_ids(d_tokens)

    assert len(q_tokens_ids) == 5
    q_seg_indices: List[List[int]] = [[0, 1, 3, 4], [2]]
    window_size = 10
    d_seg_len_list = split_window_get_length(d_tokens_ids, window_size)
    st = 0
    d_seg_indices = []
    for l in d_seg_len_list:
        ed = st + l
        d_seg_indices.append(list(range(st, ed)))
        st = ed
    text1 = SegmentedText(q_tokens_ids, q_seg_indices)
    text2 = SegmentedText(d_tokens_ids, d_seg_indices)
    inst = SegmentedInstance(text1, text2)
    return inst


def join_a_p(answer_list, problem_list):
    pid_to_p: Dict[str, RelatedEvalInstance] = index_by_fn(lambda e: e.problem_id, problem_list)
    a_p_list: List[Tuple[RelatedBinaryAnswer, RelatedEvalInstance]] = []
    for a in answer_list:
        p: RelatedEvalInstance = pid_to_p[a.problem_id]
        a_p_list.append((a, p))
    return a_p_list


class MatrixScorerIF(abc.ABC):
    @abc.abstractmethod
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        pass


class UnexpectedPolicyException(Exception):
    def __init__(self, policy_name):
        self.policy_name = policy_name

    def __str__(self):
        return f"Policy {self.policy_name} is not expected."
