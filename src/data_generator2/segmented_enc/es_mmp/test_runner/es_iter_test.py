import numpy as np
import logging
import os
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, List, Callable, OrderedDict

import numpy as np

from cache import load_pickle_from
from cpath import at_output_dir, output_path
from data_generator2.segmented_enc.es_common.es_two_seg_common import Segment1PartitionedPair
from data_generator2.segmented_enc.es_mmp.data_iter_triplets import iter_qd
from data_generator2.segmented_enc.es_mmp.pep_es_common import iter_es_data
from list_lib import assert_list_equal
from misc_lib import exist_or_mkdir, path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.chair_logging import c_log



QueryDocES = Tuple[Segment1PartitionedPair, Tuple[np.array, np.array]]

def iter_es_data_pos_neg_pair(part_no: int) -> Iterable[Tuple[QueryDocES, QueryDocES]]:
    es_data_itr = iter_es_data(part_no)
    pairs: Iterable[Segment1PartitionedPair] = iter_qd(part_no)
    paired_iter = iter(zip(pairs, es_data_itr))
    try:
        while True:
            pos_item: QueryDocES = next(paired_iter)
            pos_pair, (_pos_p1_es, _pos_p2_es) = pos_item
            pos_pair: Segment1PartitionedPair = pos_pair
            neg_item: QueryDocES = next(paired_iter)
            neg_pair, (_neg_p1_es, _neg_p2_es) = neg_item
            assert_list_equal(pos_pair.segment1.tokens, neg_pair.segment1.tokens)
            yield pos_item, neg_item
    except StopIteration:
        pass



def main():
    list(iter_es_data_pos_neg_pair(0))


if __name__ == "__main__":
    main()