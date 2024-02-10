import dataclasses
from pickle import UnpicklingError

import numpy as np

from cache import load_pickle_from
from cpath import output_path
from misc_lib import path_join
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator, Any
import os


@dataclasses.dataclass
class QDWithScoreAttn:
    query: str
    doc: str
    score: float
    attn: np.array


def get_attn2_save_dir():
    # from ce-mini-lm
    save_dir = path_join(output_path, "msmarco", "passage", "attn2_scores", )
    return save_dir


def iter_attention_data_pair(attn_save_dir, partition_no) -> Iterable[QDWithScoreAttn]:
    batch_no = 0
    while True:
        file_path = path_join(attn_save_dir, f"{partition_no}_{batch_no}")
        if os.path.exists(file_path):
            print(file_path)
            try:
                attn_data_pair: List[Tuple[str, str, float, np.array]] = load_pickle_from(file_path)
                for query, doc, score, attn in attn_data_pair:
                    yield QDWithScoreAttn(query, doc, score, attn)
            except UnpicklingError as e:
                print("Unpickling error:", e)
                print(file_path)
        else:
            break
        batch_no += 1


def iter_attention_data_pair_as_pos_neg(attn_save_dir, partition_no) -> Iterable[Tuple[QDWithScoreAttn, QDWithScoreAttn]]:
    itr = iter_attention_data_pair(attn_save_dir, partition_no)
    yield from reshape_flat_pos_neg(itr)


def reshape_flat_pos_neg(itr: Iterable[Any]) -> Iterable[Tuple[Any, Any]]:
    pos = None
    for item in itr:
        if pos is None:
            pos = item
        else:
            neg = item
            yield pos, neg
            pos = None

