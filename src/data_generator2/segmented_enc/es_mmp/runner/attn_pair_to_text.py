import csv

import numpy as np
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
from cpath import output_path
from data_generator2.segmented_enc.es_common.es_two_seg_common import PairData
from data_generator2.segmented_enc.es_mmp.pep_attn_common import iter_attention_data_pair
from misc_lib import path_join
from trainer_v2.chair_logging import c_log


def main():
    job_no = int(sys.argv[1])
    attn_data_pair: Iterable[Tuple[PairData, np.array]] = iter_attention_data_pair(job_no)
    text_save_dir = path_join(output_path, "msmarco", "passage", "mmp1_attn_text")
    text_save_path = path_join(text_save_dir, f"{job_no}.txt")
    tsv_writer = csv.writer(open(text_save_path, "w", newline=""), delimiter="\t")
    c_log.info("Begin")

    for item in attn_data_pair:
        pair, _attn = item
        row = pair.segment1, pair.segment2
        tsv_writer.writerow(row)
    c_log.info("Done")


if __name__ == "__main__":
    main()
