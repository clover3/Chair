import random

import numpy as np
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

from list_lib import left, right
from .dataloaders import DataLoaderWrapper
from ..utils.utils import rename_keys
from transformers.tokenization_utils_base import BatchEncoding, EncodingFast

from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

@dataclass
class PEPTrainBlock:
    input_ids: list[int]
    token_type_ids: list[int]
    attn_indices: list[int]
    cross_attn_indices: list[int]
    doc_cls_indices: list[int]


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
    # attn = np.zeros([seq_len, seq_len], dtype=int).tolist()
    attn = np.zeros([seq_len, seq_len], dtype=int)

    n_seg = len(attn_indices) // 2
    assert len(attn_indices) % 2 == 0
    for i in range(n_seg):
        st, ed = attn_indices[i * 2], attn_indices[i * 2 + 1]
        if st < seq_len and ed < seq_len:
            attn[st:ed, st:ed] = 1
            # for jy in range(st, ed):
            #     for jx in range(st, ed):
            #         print(jy, jx)
            #         attn[jy][jx] = 1

    n_block = len(cross_attn_indices) // 4
    assert len(cross_attn_indices) % 4 == 0
    for i in range(n_block):
        st, ed, st_to, ed_to = cross_attn_indices[i * 4: (i + 1) * 4]
        if st < seq_len and ed < seq_len and st_to < seq_len and ed_to < seq_len:
            attn[st:ed, st_to:ed_to] = 1

            # for jy in range(st, ed):
            #     for jx in range(st_to, ed_to):
            #         try:
            #             attn[jy][jx] = 1
            #         except IndexError:
            #             print(jy, jx, len(attn))

    return attn


def set_as_mask(doc_cls_indices, seq_len):
    mask = [0] * seq_len
    for i in doc_cls_indices:
        if i < seq_len:
            mask[i] = 1
    return mask


def pad_truncate(ids_list: list[list[int]], max_length, pad_token_id):
    longest_seq = max(map(len, ids_list))
    target_len = min(longest_seq, max_length)
    for idx, item in enumerate(ids_list):
        if len(item) > target_len:
            ids_list[idx] = item[:target_len]
        else:
            n_pad = target_len - len(item)
            ids_list[idx] = item + [pad_token_id] * n_pad

    for item in ids_list:
        assert len(item) == target_len
    return ids_list


class PartitionEncoder:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_token_id = tokenizer.mask_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

    def partition_query(self, query: EncodingFast) -> list[int]:
        bounds: list[int] = get_word_boundaries(query.word_ids)
        st = random.randint(0, len(bounds) - 1)
        if st + 1 <= len(bounds) - 1:
            ed = random.randint(st + 1, len(bounds) - 1)
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
            for i in range(len(indices_doc) - 1):
                st = indices_doc[i]
                ed = indices_doc[i + 1]
                yield d_rep.ids[st: ed]

        def get_pep_train_block(qt) -> PEPTrainBlock:
            attn_indices = [0]
            qd_seg = [CLS] + qt
            attn_indices.append(len(qd_seg))
            n_q_seg_tokens = len(qd_seg)
            st_to = attn_indices[0]
            ed_to = attn_indices[1]
            cross_attn_indices = []
            doc_cls_indices = []

            for dseg in iter_dseg_items():
                st = len(qd_seg)
                doc_cls_indices.append(st)
                qd_seg.append(CLS)
                qd_seg += dseg
                ed = len(qd_seg)
                attn_indices.append(st)
                attn_indices.append(ed)
                cross_attn_indices.extend([st, ed, st_to, ed_to])

            qd_seg.append(SEP)
            n_d_seg_tokens = len(qd_seg) - n_q_seg_tokens
            token_type_ids = [0] * n_q_seg_tokens + [1] * n_d_seg_tokens
            item: PEPTrainBlock = PEPTrainBlock(
                qd_seg, token_type_ids, attn_indices, cross_attn_indices, doc_cls_indices)
            return item

        return get_pep_train_block(qt1), get_pep_train_block(qt2)
        # Output:
        # - token_ids which concatenate all query terms and document terms
        # - Indices for building attention matrix
        # -     Self attention range (st1, ed1, st2, ed2 ... )
        # -     Cross attention range (st_from, ed_from, st_to, ed_to) * repeat

    def as_batch_arrays(
            self,
            items: list[tuple[PEPTrainBlock, PEPTrainBlock]]) -> dict[str, np.array]:
        left_blocks = left(items)
        right_blocks = right(items)
        block_d = {
            'left': left_blocks,
            'right': right_blocks,
        }

        def pad_fn(ids_list: list[list[int]]):
            return pad_truncate(ids_list, self.max_length, self.tokenizer.pad_token_id)

        ret = {}
        for role in ["left", "right"]:
            blocks = block_d[role]
            input_ids_list: list[list[int]] = [e.input_ids for e in blocks]
            token_type_ids_list = [e.token_type_ids for e in blocks]
            input_ids = pad_fn(input_ids_list)
            token_type_ids = pad_fn(token_type_ids_list)

            seq_len = len(input_ids[0])
            attn_list = [build_attn_array(
                e.attn_indices, e.cross_attn_indices, seq_len) for e in blocks]

            doc_cls_indices = [set_as_mask(e.doc_cls_indices, seq_len) for e in blocks]

            ret[f'{role}_input_ids'] = np.array(input_ids)
            ret[f'{role}_token_type_ids'] = np.array(token_type_ids)
            ret[f'{role}_attention_mask'] = np.array(attn_list)
            ret[f'{role}_doc_cls_indices'] = np.array(doc_cls_indices)
        return ret


# encoded_pos has sequence like
# Query is partitioned into two segments and concatenated
# Seg_Query1 = [CLS] [Q_Token1] [Q_Token2] [SEP]
# Seg_Query2 = [CLS] [Q_Token3] ... [Q_Token_N] [SEP]
# DocSeg = [SEP] [D_Tokens1] [D_Tokens2] [D_Tokens3] [SEP] [D_Tokens4] [D_Tokens5] [SEP] [D_Tokens6] [SEP]
# Seg1 = Seg_Query1 + DocSeg
# Seg2 = Seg_Query2 + DocSeg
class PEPPairsDataLoaderDistil(DataLoaderWrapper):
    """Siamese encoding (query and document independent)
    train mode (pairs)
    """

    def __init__(self, tokenizer_type, max_length, **kwargs):
        super(PEPPairsDataLoaderDistil, self).__init__(tokenizer_type, max_length, **kwargs)
        self.encoder: PartitionEncoder = PartitionEncoder(self.tokenizer, max_length)

    def collate_fn(self, batch):
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


class PEPPairsDataLoader(DataLoaderWrapper):
    def __init__(self, tokenizer_type, max_length, **kwargs):
        super(PEPPairsDataLoader, self).__init__(tokenizer_type, max_length, **kwargs)
        self.encoder: PartitionEncoder = PartitionEncoder(self.tokenizer, max_length)

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 3 (text) items (q, d_pos, d_neg)
        """
        q, d_pos, d_neg = zip(*batch)
        n_item = len(q)

        q_rep = self.tokenizer(q)
        indice_q_seg: list[list[int]] = [self.encoder.partition_query(q_rep[i]) for i in range(n_item)]
        doc_d = {
            "pos": d_pos,
            "neg": d_neg,
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

        return {k: torch.tensor(v) for k, v in sample.items()}


class PEPGroupingDataLoader(DataLoaderWrapper):
    def __init__(self, tokenizer_type, max_length, **kwargs):
        super(PEPGroupingDataLoader, self).__init__(tokenizer_type, max_length, **kwargs)
        self.encoder: PartitionEncoder = PartitionEncoder(self.tokenizer, max_length)
        self.q_indices_d: dict[str, list[int]] = {}

    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (q, d)
        """
        q, d = zip(*batch)
        n_item = len(batch)
        q_rep = self.tokenizer(q)
        d_rep = self.tokenizer(d)

        entries = []
        for i in range(n_item):
            query: str = q[i]
            try:
                q_indices = self.q_indices_d[query]
            except KeyError:
                q_indices = self.encoder.partition_query(q_rep[i])
                self.q_indices_d[query] = q_indices

            indices_doc: list[int] = self.encoder.partition_doc(d_rep[i])
            encoded = self.encoder.combine(q_rep[i], d_rep[i], q_indices, indices_doc)
            entries.append(encoded)

        encoded = self.encoder.as_batch_arrays(entries)
        return {k: torch.tensor(v) for k, v in encoded.items()}


class QDListDataset(Dataset):
    def __init__(self, qd_list: list[tuple[str, str]]):
        self.qd_list = qd_list

    def __getitem__(self, idx):
        return self.qd_list[idx]

    def __len__(self):
        return len(self.qd_list)