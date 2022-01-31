from collections import defaultdict
from typing import NamedTuple, List, Iterator, Dict, Tuple, Callable, Any

import numpy as np


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