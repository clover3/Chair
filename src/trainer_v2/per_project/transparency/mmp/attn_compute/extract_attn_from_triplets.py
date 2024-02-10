import pickle
import sys

from misc_lib import path_join

import numpy as np
from typing import List, Tuple
from dataset_specific.msmarco.passage.path_helper import train_triples_small_partition_iter
from misc_lib import batch_iter_from_entry_iter
from ptorch.cross_encoder.attention_extractor import AttentionExtractor
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.attn_compute.iter_attn import get_attn2_save_dir


def main():
    job_no = int(sys.argv[1])
    # This contains about 40,000,000 lines
    inf_batch_size = 100
    save_batch_size = 1000  # 6sec per batch

    lines = train_triples_small_partition_iter(job_no)
    save_dir = get_attn2_save_dir()
    def triplet_iter():
        for l in lines:
            yield l.split("\t")

    extractor = AttentionExtractor()
    for idx, batch in enumerate(batch_iter_from_entry_iter(triplet_iter(), save_batch_size)):
        c_log.info("Batch %d", idx)
        todo: List[Tuple[str, str]] = []
        for q, dp, dn in batch:
            todo.append((q, dp))
            todo.append((q, dn))

        out_items: List[Tuple[str, str, float, np.array]] = extractor.predict(todo)
        c_log.info("Saving pickle")
        save_path = path_join(save_dir, f"{job_no}_{idx}")
        pickle.dump(out_items, open(save_path, "wb"))


if __name__ == "__main__":
    main()
