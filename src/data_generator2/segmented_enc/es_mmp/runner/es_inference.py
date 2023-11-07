import os.path
import pickle
import sys
from typing import Iterable, Callable, List

import numpy as np
from transformers import AutoTokenizer

from data_generator2.segmented_enc.es_common.es_two_seg_common import Segment1PartitionedPair, BothSegPartitionedPair
from data_generator2.segmented_enc.es_common.partitioned_encoder import get_both_seg_partitioned_to_input_ids2
from data_generator2.segmented_enc.es_mmp.data_iter_triplets import iter_qd
from list_lib import apply_batch
from misc_lib import path_join, TimeEstimator
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.inference import BERTInferenceHelper
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def parse_score(e: Segment1PartitionedPair, score_pair: np.array) -> List[np.array]:
    segment1 = e.segment1
    segment2 = e.segment2

    output = []
    for i in [0, 1]:
        seg1_st = 1
        seg1_ed = seg1_st + len(segment1.get(i))
        seg2_st = seg1_ed + 1
        seg2_ed = seg2_st + len(segment2)
        score: np.array = score_pair[i]
        seg2_scores = score[seg2_st: seg2_ed]
        output.append(seg2_scores)
    return output


def do_for_partition(score_fn, save_dir, partition_no):
    def get_save_path_fn(batch_idx):
        return path_join(save_dir, f"{partition_no}_{batch_idx}")

    data_size = 1000000
    pair_iter: Iterable[Segment1PartitionedPair] = iter_qd(partition_no)
    save_batch_size = 1024 * 8
    num_batch = data_size // save_batch_size
    ticker = TimeEstimator(num_batch)
    for batch_idx, batch in enumerate(apply_batch(pair_iter, save_batch_size)):
        ticker.tick()

        save_path = get_save_path_fn(batch_idx)
        if os.path.exists(save_path):
            continue

        scores: np.array = score_fn(batch)
        if not len(scores) == len(batch):
            print("len scores", len(scores))
            print(len("batch"))

        save_payload = []
        for pair, score_pair in zip(batch, scores):
            sliced_scores = parse_score(pair, score_pair)
            save_payload.append(sliced_scores)

        pickle.dump(save_payload, open(save_path, "wb"))
        c_log.info("Saved at %s", save_path)


def get_es_model(model_save_path) -> Callable[[Iterable[Segment1PartitionedPair]], np.array]:
    max_seq_length = 256
    strategy = get_strategy()
    inf_helper = BERTInferenceHelper(model_save_path, max_seq_length, strategy)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    EncoderType = Callable[[BothSegPartitionedPair], Iterable[tuple[list, list]]]
    encoder_iter: EncoderType = get_both_seg_partitioned_to_input_ids2(tokenizer, max_seq_length)

    def score_fn(payload: Iterable[Segment1PartitionedPair]) -> np.array:
        c_log.debug("score_fn entry")
        payload1: Iterable[BothSegPartitionedPair] = map(BothSegPartitionedPair.from_seg1_partitioned_pair, payload)
        c_log.debug("score_fn 1")
        payload2: list[tuple[list, list]] = []
        for item in payload1:
            payload2.extend(encoder_iter(item))
        c_log.debug("score_fn 2")

        ret: np.array = inf_helper.predict(payload2)  # [N*2, L, 1]
        B2, L, _ = ret.shape
        B = B2 // 2
        ret = np.reshape(ret, [B, 2, L])
        c_log.debug("score_fn 3")
        print("inf_helper outputs", ret.shape)
        return ret

    return score_fn


def main(model_save_path, save_dir, job_no):
    # c_log.setLevel(logging.DEBUG)
    score_fn = get_es_model(model_save_path)
    partition_per_job = 1
    st = job_no * partition_per_job
    ed = st + partition_per_job
    for partition_no in range(st, ed):
        try:
            print(f"Partition {partition_no}")
            do_for_partition(score_fn, save_dir, partition_no)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    model_save_path = sys.argv[1]
    save_dir = sys.argv[2]
    job_no = int(sys.argv[3])

    with JobContext(f"ESInf {job_no}"):
        main(model_save_path, save_dir, job_no)
