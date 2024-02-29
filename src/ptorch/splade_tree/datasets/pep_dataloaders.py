import random

import numpy as np
from dataclasses import dataclass
import torch

from list_lib import left, right
from .dataloaders import DataLoaderWrapper
from ..utils.utils import rename_keys
from transformers.tokenization_utils_base import BatchEncoding, EncodingFast


@dataclass
class PEPTrainBlock:
    input_ids: list[int]
    token_type_ids: list[int]
    attn_indices: list[int]
    cross_attn_indices: list[int]


EncodedItem = NotImplemented


def get_word_boundaries(word_ids):
    prev_word_id = None
    bounds = []
    for idx, item in enumerate(word_ids):
        if prev_word_id != item:
            bounds.append(idx)

        prev_word_id = item
    return bounds


def build_attn_array(attn_indices, cross_attn_indices, seq_len):
    attn = [[0] * seq_len] * seq_len

    n_seg = len(attn_indices) // 2

    for i in range(n_seg):
        st, ed = attn_indices[i * 2], attn_indices[i * 2 + 1]
        for jy in range(st, ed):
            for jx in range(st, ed):
                attn[jy][jx] = 1

    n_block = len(cross_attn_indices) // 4
    assert len(cross_attn_indices) % 4 == 0

    for i in range(n_block):
        st, ed, st_to, ed_to = attn_indices[i * 4: (i+1) * 4]
        for jy in range(st, ed):
            for jx in range(st_to, ed_to):
                attn[jy][jx] = 1
    return attn


class PartitionEncoder:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_token_id = tokenizer.mask_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

    def partition_query(self, query: EncodingFast) -> list[int]:
        bounds = get_word_boundaries(query.word_ids)
        st = random.randint(0, len(bounds))
        if st + 1 <= len(bounds):
            ed = random.randint(st + 1, len(bounds))
        else:
            ed = st

        return [0, bounds[st], bounds[ed], len(bounds)]

    def partition_doc(self, doc: EncodingFast) -> list[int]:
        bounds = get_word_boundaries(doc.word_ids)

        out_indices = []
        cursor = 0
        while cursor < len(bounds):
            n_word = random.randint(1, 3)
            segment = bounds[cursor:cursor + n_word]
            st = segment[0]
            ed = segment[-1]

            if st < ed:
                out_indices.append(st)

            cursor += n_word
        out_indices.append(bounds[-1])
        return out_indices

    def combine(
            self,
            q_rep: EncodingFast,
            d_rep: EncodingFast,
            indices_q_seg: list[int],
            indices_doc: list[int]) -> tuple[PEPTrainBlock, PEPTrainBlock]:
        MASK_ID = self.mask_token_id
        CLS = self.cls_token_id
        SEP = self.sep_token_id

        q_st, q_ed, seg_st, seg_ed = indices_q_seg
        qt1 = q_rep.ids[seg_st:seg_ed]
        qt2 = q_rep.ids[q_st:seg_st] + [MASK_ID] + q_rep.ids[seg_ed: q_ed]

        def iter_dseg_items():
            for i in range(len(indices_doc)-1):
                st = indices_doc[i]
                ed = indices_doc[i+1]
                yield d_rep.ids[st: ed]

        def get_pep_train_block(qt) -> PEPTrainBlock:
            attn_indices = [0]
            qd_seg = [CLS] + qt
            attn_indices.append(len(qd_seg))
            n_q_seg_tokens = len(qd_seg)
            st_to = attn_indices[0]
            ed_to = attn_indices[1]
            cross_attn_indices = []

            for dseg in iter_dseg_items():
                st = len(qd_seg)
                qd_seg.append(SEP)
                qd_seg += dseg
                ed = len(qd_seg)
                attn_indices.append(st)
                attn_indices.append(ed)
                cross_attn_indices.extend([st, ed, st_to, ed_to])

            qd_seg.append(SEP)
            n_d_seg_tokens = len(qd_seg) - n_q_seg_tokens
            token_type_ids = [0] * n_q_seg_tokens + [1] * n_d_seg_tokens
            item: PEPTrainBlock = PEPTrainBlock(
                qd_seg, token_type_ids, attn_indices, cross_attn_indices)
            return item

        return get_pep_train_block(qt1), get_pep_train_block(qt2)
        # Output:
        # - token_ids which concatenate all query terms and document terms
        # - Indices for building attention matrix
        # -     Self attention range (st1, ed1, st2, ed2 ... )
        # -     Cross attention range (st_from, ed_from, st_to, ed_to) * repeat

    def as_batch_arrays(self,
            items: list[tuple[PEPTrainBlock, PEPTrainBlock]]) -> dict[str, np.array]:
        left_blocks = left(items)
        right_blocks = right(items)
        block_d = {
            'left': left_blocks,
            'right': right_blocks,
        }

        def pad_fn(ids_list):
            return self.tokenizer.pad(
                ids_list,
                add_special_tokens=False,
                padding="longest",
                truncation="longest_first",
                max_length=self.max_length,
                return_attention_mask=False)

        ret = {}
        for role in ["left", "right"]:
            blocks = block_d[role]
            input_ids_list: list[list[int]] = [e.input_ids for e in blocks]
            token_type_ids_list = [e.token_type_ids for e in blocks]
            seq_len = len(input_ids_list[0])
            attn_list = [build_attn_array(
                e.attn_indices, e.cross_attn_indices, seq_len) for e in blocks]

            b1: BatchEncoding = pad_fn(input_ids_list)
            input_ids = b1.input_ids
            b2: BatchEncoding = pad_fn(token_type_ids_list)
            token_type_ids = b2.input_ids
            ret[f'{role}_input_ids'] = input_ids
            ret[f'{role}_token_type_ids'] = token_type_ids
            ret[f'{role}_attn_mask'] = attn_list
        return ret


# encoded_pos has sequence like
# Query is partitioned into two segments and concatenated
# Seg_Query1 = [CLS] [Q_Token1] [Q_Token2] [SEP]
# Seg_Query2 = [CLS] [Q_Token3] ... [Q_Token_N] [SEP]
# DocSeg = [SEP] [D_Tokens1] [D_Tokens2] [D_Tokens3] [SEP] [D_Tokens4] [D_Tokens5] [SEP] [D_Tokens6] [SEP]
# Seg1 = Seg_Query1 + DocSeg
# Seg2 = Seg_Query2 + DocSeg
class PEPPairsDataLoader(DataLoaderWrapper):
    """Siamese encoding (query and document independent)
    train mode (pairs)
    """
    def __init__(self, tokenizer_type, max_length, **kwargs):
        super(PEPPairsDataLoader, self).__init__(tokenizer_type, max_length, **kwargs)
        self.encoder: PartitionEncoder = PartitionEncoder(self.tokenizer, max_length)

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 3 (text) items (q, d_pos, d_neg)
        """
        q, d_pos, d_neg, s_pos, s_neg = zip(*batch)
        n_item = len(q)

        q_rep = self.tokenizer(q)
        indice_q_seg: list[list[int]] = [self.encoder.partition_query(q_rep[i]) for i in range(n_item)]
        doc_d = {
            "pos": d_pos,
            "neg": d_neg,
        }
        score_d = {
            "pos": s_pos,
            "neg": s_neg,
        }

        sample = {}
        for role in ["pos", "neg"]:
            d_rep = self.tokenizer(doc_d[role])
            entries = []
            for i in range(n_item):
                indices_doc: list[int] = self.encoder.partition_doc(d_rep[i])
                encoded = self.encoder.combine(q_rep[i], d_rep[i], indice_q_seg[i], indices_doc)
                entries.append(encoded)

            doc_encoded = self.encoder.as_batch_arrays(entries)
            sample.update(rename_keys(doc_encoded, role))
            sample[f"teacher_{role}_score"] = score_d[role]

        return {k: torch.tensor(v) for k, v in sample.items()}



