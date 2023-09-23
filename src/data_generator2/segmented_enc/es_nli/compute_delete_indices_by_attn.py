from typing import List, Iterable

import numpy as np

from data_generator2.segmented_enc.es_nli.common import PHSegmentedPair, HSegmentedPair
from misc_lib import TimeEstimator
from trainer_v2.custom_loop.attention_helper.evidence_selector_0 import get_delete_indices


def compute_attn_sel_delete_indices(
        attn_extractor,
        itr: Iterable[HSegmentedPair],
        data_size,
        external_batch_size) -> Iterable[PHSegmentedPair]:

    def iter_batches():
        batch = []
        for e in itr:
            batch.append(e)
            if len(batch) == external_batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    ticker = TimeEstimator(data_size, sample_size=100)
    for e_batch in iter_batches():
        payload = []
        for e in e_batch:
            payload.append((e.p_tokens, e.h_tokens))

        res = attn_extractor.predict_list(payload)

        for i, _ in enumerate(payload):
            e = e_batch[i]
            attn_score: np.array = res[i]
            delete_indices_list = get_delete_indices(attn_score, e)

            e_out = PHSegmentedPair(e.p_tokens, e.h_tokens, e.st, e.ed,
                                    delete_indices_list[0], delete_indices_list[1],
                                    e.nli_pair)
            yield e_out
            ticker.tick()


def iter_batches(itr, batch_size):
    batch = []
    for e in itr:
        batch.append(e)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_attn_sel_delete_indices_itr(
        attn_extractor,
        itr: List[HSegmentedPair],
        external_batch_size) -> Iterable[PHSegmentedPair]:

    for e_batch in iter_batches(itr, external_batch_size):
        payload = []
        for e in e_batch:
            payload.append((e.p_tokens, e.h_tokens))

        res = attn_extractor.predict_list(payload)

        for i, _ in enumerate(payload):
            e = e_batch[i]
            attn_score: np.array = res[i]
            delete_indices_list = get_delete_indices(attn_score, e)
            e_out = PHSegmentedPair(e.p_tokens, e.h_tokens, e.st, e.ed,
                                    delete_indices_list[0], delete_indices_list[1],
                                    e.nli_pair)
            yield e_out
