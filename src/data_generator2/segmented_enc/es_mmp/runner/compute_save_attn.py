import pickle
import sys
from typing import Iterable, Tuple

import numpy as np

from data_generator2.segmented_enc.es_mmp.iterate_mmp import iter_qd_sample_as_pair_data
from data_generator2.segmented_enc.es_common.es_two_seg_common import PairData
from misc_lib import TimeEstimator, path_join
from trainer_v2.custom_loop.attention_helper.attention_extractor_hf import load_mmp1_attention_extractor
from trainer_v2.custom_loop.attention_helper.attention_extractor_interface import AttentionExtractor
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def compute_attn_save(
        attn_extractor,
        itr: Iterable[PairData],
        data_size,
        external_batch_size,
        get_save_path_fn):

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
    for batch_idx, e_batch in enumerate(iter_batches()):
        payload = []
        for e in e_batch:
            payload.append((e.segment1, e.segment2))

        res = attn_extractor.predict_list(payload)
        output = []

        for i, _ in enumerate(payload):
            e = e_batch[i]
            attn_score: np.array = res[i]
            out_row: Tuple[PairData, np.array] = e, attn_score
            ticker.tick()
            output.append(out_row)
        pickle.dump(output, open(get_save_path_fn(batch_idx), "wb"))


def do_for_partition(attn_extractor, partition_no, save_dir):
    data_size = 30000
    save_batch_size = 1024
    pair_iter = iter_qd_sample_as_pair_data(partition_no)

    def get_save_path_fn(batch_idx):
        return path_join(save_dir, f"{partition_no}_{batch_idx}")

    compute_attn_save(
        attn_extractor, pair_iter, data_size, save_batch_size, get_save_path_fn)


def main(job_no, save_dir):
    attn_extractor: AttentionExtractor = load_mmp1_attention_extractor()
    partition_per_job = 10
    st = job_no * partition_per_job
    ed = st + partition_per_job
    for partition_no in range(st, ed):
        try:
            print(f"Partition {partition_no}")
            do_for_partition(attn_extractor, partition_no, save_dir)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    job_no = int(sys.argv[1])
    save_dir = sys.argv[2]
    strategy = get_strategy(False, "")
    with strategy.scope():
        main(job_no, save_dir)
