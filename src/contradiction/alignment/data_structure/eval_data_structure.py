import abc
from typing import NamedTuple, List, Dict, Tuple, Any, Optional

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import index_by_fn
from tlm.data_gen.doc_encode_common import split_window_get_length
from contradiction.alignment.data_structure.related_eval_instance import RelatedEvalInstance

ContributionSummaryDict = Dict[str, List[float]]


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
