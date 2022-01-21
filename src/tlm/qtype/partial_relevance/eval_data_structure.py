from collections import defaultdict
from typing import NamedTuple, List, Iterator, Dict, Tuple, Callable, Any

import numpy as np

from cache import named_tuple_to_json
from data_generator.tokenizer_wo_tf import pretty_tokens, ids_to_text
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo


class ContributionSummary(NamedTuple):
    table: List[List[float]]


ContributionSummaryDict = Dict[str, List[float]]


class QDSegmentedInstance(NamedTuple):
    text1_tokens_ids: List # This is supposed to be shorter one. (Query)
    text2_tokens_ids: List # This is longer one (Document)
    label: int
    seg2_len: int
    n_q_segs: int
    d_seg_len_list: List[int]
    q_seg_indices: List[List[int]]

    def enum_seg_indice_pairs(self):
        for q_seg_idx in range(self.n_q_segs):
            for d_seg_idx in range(self.seg2_len):
                yield q_seg_idx, d_seg_idx

    def get_empty_doc_mask(self):
        return np.zeros([self.seg2_len], np.int)

    def get_drop_mask(self, q_seg_idx, d_seg_idx) -> np.ndarray:
        drop_mask_per_q_seg = self.get_empty_doc_mask()
        drop_mask_per_q_seg[d_seg_idx] = 1
        drop_mask = [self.get_empty_doc_mask() for _ in range(self.n_q_segs)]
        drop_mask[q_seg_idx] = drop_mask_per_q_seg
        drop_mask = np.stack(drop_mask)
        return drop_mask

    def get_empty_mask(self) -> np.ndarray:
        return np.stack([self.get_empty_doc_mask() for _ in range(self.n_q_segs)])

    def enum_token_idx_from_q_seg_idx(self, q_seg_idx) -> Iterator[int]:
        yield from self.q_seg_indices[q_seg_idx]

    def enum_token_idx_from_d_seg_idx(self, d_seg_idx) -> Iterator[int]:
        n_q_tokens = len(self.text1_tokens_ids)
        base = 1 + n_q_tokens + 1
        start = base
        for d_seg_idx2 in range(0, d_seg_idx):
            start += self.d_seg_len_list[d_seg_idx2]

        for idx in range(self.d_seg_len_list[d_seg_idx]):
            yield start + idx

    def translate_mask(self, drop_mask) -> Dict[Tuple[int, int], int]:
        new_mask = {}
        for q_seg_idx in range(self.n_q_segs):
            for d_seg_idx in range(self.seg2_len):
                v = drop_mask[q_seg_idx, d_seg_idx]
                if v:
                    for q_token_idx in self.enum_token_idx_from_q_seg_idx(q_seg_idx):
                        for d_token_idx in self.enum_token_idx_from_d_seg_idx(d_seg_idx):
                            k = q_token_idx, d_token_idx
                            new_mask[k] = int(v)
        return new_mask

    def accumulate_over(self, raw_scores, accumulate_method: Callable[[List[float]], float]):
        scores_d = defaultdict(list)
        for q_seg_idx in range(self.n_q_segs):
            for d_seg_idx in range(self.seg2_len):
                key = q_seg_idx, d_seg_idx
                for q_token_idx in self.enum_token_idx_from_q_seg_idx(q_seg_idx):
                    for d_token_idx in self.enum_token_idx_from_d_seg_idx(d_seg_idx):
                        v = raw_scores[q_token_idx, d_token_idx]
                        scores_d[key].append(v)

        out_d = {}
        for q_seg_idx in range(self.n_q_segs):
            for d_seg_idx in range(self.seg2_len):
                key = q_seg_idx, d_seg_idx
                scores = scores_d[key]
                out_d[key] = accumulate_method(scores)
        return out_d

    def score_d_to_table(self, contrib_score_d: Dict[Tuple[int, int], Any]):
        return self._score_d_to_table(contrib_score_d)

    def _score_d_to_table(self, contrib_score_table):
        table = []
        for q_seg_idx in range(self.n_q_segs):
            row = []
            for d_seg_idx in range(self.seg2_len):
                key = q_seg_idx, d_seg_idx
                row.append(contrib_score_table[key])
            table.append(row)
        return table

    def score_np_table_to_table(self, contrib_score_table):
        return self._score_d_to_table(contrib_score_table)

    def get_seg1_len(self):
        return self.n_q_segs

    def get_seg2_len(self):
        return self.seg2_len


class SegmentedText(NamedTuple):
    tokens_ids: List
    seg_token_indices: List[List[int]]

    def get_seg_len(self):
        return len(self.seg_token_indices)

    def enum_seg_idx(self):
        yield from range(self.get_seg_len())

    def get_empty_seg_mask(self):
        return np.zeros([self.get_seg_len()], np.int)

    def enum_token_idx_from_seg_idx(self, seg_idx) -> Iterator[int]:
        yield from self.seg_token_indices[seg_idx]

    def get_token_idx_as_head_tail(self, seg_idx) -> Tuple[List[int], List[int]]:
        indice = self.seg_token_indices[seg_idx]
        prev_idx = None
        split_idx = len(indice)
        for j, idx in enumerate(indice):
            if prev_idx is not None:
                if prev_idx == idx-1:
                    pass
                else:
                    split_idx = j

        return indice[:split_idx], indice[split_idx:]

    def get_tokens_for_seg(self, seg_idx):
        return [self.tokens_ids[i] for i in self.seg_token_indices[seg_idx]]

    def get_dropped_text(self, drop_indices):
        new_seg = []
        new_seg_indices = []
        offset = 0
        for seg_idx in range(self.get_seg_len()):
            if seg_idx in drop_indices:
                offset = offset - len(self.seg_token_indices[seg_idx])
            else:
                for i in self.enum_token_idx_from_seg_idx(seg_idx):
                    new_seg.append(self.tokens_ids[i])

                new_indices = [idx + offset for idx in self.enum_token_idx_from_seg_idx(seg_idx)]
                new_seg_indices.append(new_indices)
        return SegmentedText(new_seg, new_seg_indices)

    def to_json(self):
        return named_tuple_to_json(self)

    @classmethod
    def from_json(cls, j):
        return SegmentedText(j['tokens_ids'], j['seg_token_indices'])


def print_segment_text(tokenizer, text: SegmentedText):
    for i in range(text.get_seg_len()):
        tokens = text.get_tokens_for_seg(i)
        s = ids_to_text(tokenizer, tokens)
        print("{}:\t{}".format(i, s))


class SegmentedInstanceOld(NamedTuple):
    text1_tokens_ids: List  # This is supposed to be shorter one. (Query)
    text2_tokens_ids: List  # This is longer one (Document)
    text1_seg_indices: List[List[int]]
    text2_seg_indices: List[List[int]]
    score: float

    def enum_seg_indice_pairs(self):
        for seg1_idx in range(self.get_seg1_len()):
            for seg2_idx in range(self.get_seg2_len()):
                yield seg1_idx, seg2_idx

    def get_seg1_len(self):
        return len(self.text1_seg_indices)

    def get_seg2_len(self):
        return len(self.text2_seg_indices)

    def get_empty_seg2_mask(self):
        return np.zeros([self.get_seg2_len()], np.int)

    def get_drop_mask(self, seg1_idx, seg2_idx) -> np.ndarray:
        drop_mask_per_q_seg = self.get_empty_seg2_mask()
        drop_mask_per_q_seg[seg2_idx] = 1
        drop_mask = [self.get_empty_seg2_mask() for _ in range(self.get_seg1_len())]
        drop_mask[seg1_idx] = drop_mask_per_q_seg
        drop_mask = np.stack(drop_mask)
        return drop_mask

    def get_empty_mask(self) -> np.ndarray:
        return np.stack([self.get_empty_seg2_mask() for _ in range(self.get_seg1_len())])

    def enum_token_idx_from_seg1_idx(self, seg_idx) -> Iterator[int]:
        yield from self.text2_seg_indices[seg_idx]

    def enum_token_idx_from_seg2_idx(self, d_seg_idx) -> Iterator[int]:
        yield from self.text2_seg_indices[d_seg_idx]

        base = 1 + self.get_seg1_len() + 1
        start = base
        for d_seg_idx2 in range(0, d_seg_idx):
            start += len(self.text2_seg_indices[d_seg_idx2])

        for idx in self.text2_seg_indices[d_seg_idx]:
            yield start + idx

    def translate_mask(self, drop_mask) -> Dict[Tuple[int, int], int]:
        new_mask = {}
        for q_seg_idx in range(self.get_seg1_len()):
            for d_seg_idx in range(self.get_seg2_len()):
                v = drop_mask[q_seg_idx, d_seg_idx]
                if v:
                    for q_token_idx in self.enum_token_idx_from_seg1_idx(q_seg_idx):
                        for d_token_idx in self.enum_token_idx_from_seg2_idx(d_seg_idx):
                            k = q_token_idx, d_token_idx
                            new_mask[k] = int(v)
        return new_mask

    def accumulate_over(self, raw_scores, accumulate_method: Callable[[List[float]], float]):
        scores_d = defaultdict(list)
        for q_seg_idx in range(self.get_seg1_len()):
            for d_seg_idx in range(self.get_seg2_len()):
                key = q_seg_idx, d_seg_idx
                for q_token_idx in self.enum_token_idx_from_seg1_idx(q_seg_idx):
                    for d_token_idx in self.enum_token_idx_from_seg2_idx(d_seg_idx):
                        v = raw_scores[q_token_idx, d_token_idx]
                        scores_d[key].append(v)

        out_d = {}
        for q_seg_idx in range(self.get_seg1_len()):
            for d_seg_idx in range(self.get_seg2_len()):
                key = q_seg_idx, d_seg_idx
                scores = scores_d[key]
                out_d[key] = accumulate_method(scores)
        return out_d

    def score_d_to_table(self, contrib_score_d: Dict[Tuple[int, int], Any]):
        return self._score_d_to_table(contrib_score_d)

    def _score_d_to_table(self, contrib_score_table):
        table = []
        for q_seg_idx in range(self.get_seg1_len()):
            row = []
            for d_seg_idx in range(self.get_seg2_len()):
                key = q_seg_idx, d_seg_idx
                row.append(contrib_score_table[key])
            table.append(row)
        return table

    def score_np_table_to_table(self, contrib_score_table):
        return self._score_d_to_table(contrib_score_table)

    def get_seg2_dropped_instances(self, drop_indices):
        new_seg2 = []
        new_seg2_indices = []
        for seg_idx in range(self.get_seg2_len()):
            if seg_idx in drop_indices:
                pass
            else:
                for i in self.enum_token_idx_from_seg2_idx(seg_idx):
                    new_seg2.append(i)
                new_seg2_indices.append(list(self.enum_token_idx_from_seg2_idx(seg_idx)))
        return SegmentedInstanceOld(self.text1_tokens_ids,
                                 new_seg2,
                                 self.text1_seg_indices,
                                 new_seg2_indices,
                                 self.score
                                 )


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


def rei_to_text(tokenizer, rei: RelatedEvalInstance):
    def seg_to_text(segment: SegmentedText) -> str:
        ids: List[int] = segment.tokens_ids
        tokens = tokenizer.convert_ids_to_tokens(ids)
        return pretty_tokens(tokens, True)

    seg1_text = seg_to_text(rei.seg_instance.text1)
    seg2_text = seg_to_text(rei.seg_instance.text2)
    return f"RelatedEvalInstance({rei.problem_id}, {seg1_text})\n" \
           + "Doc: " + seg2_text


class RelatedEvalAnswer(NamedTuple):
    problem_id: str
    contribution: ContributionSummary

