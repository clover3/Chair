from typing import Tuple, Dict

from arg.perspectives.pc_rel.collect_score import combine_pc_rel_with_cpid
from arg.perspectives.types import DataID, Logits, CPIDPair
from cache import load_from_pickle, save_to_pickle
from cpath import pjoin, output_path


def save_for_train():
    info = load_from_pickle("pc_rel_info_all")
    prediction_path = pjoin(output_path, "pc_rel")
    rel_info: Dict[DataID, Tuple[CPIDPair, Logits, Logits]] = combine_pc_rel_with_cpid(prediction_path, info)
    save_to_pickle(rel_info, "pc_rel_with_cpid")


def save_for_dev():
    info = load_from_pickle("pc_rel_dev_info_all")
    prediction_path = pjoin(output_path, "pc_rel_dev")
    rel_info: Dict[DataID, Tuple[CPIDPair, Logits, Logits]] = combine_pc_rel_with_cpid(prediction_path, info)
    save_to_pickle(rel_info, "pc_rel_dev_with_cpid")


def pc_rel_dev_with_cpid():
    rel_info = load_from_pickle("pc_rel_dev_with_cpid")
    keys = list(rel_info.keys())
    print("num keys", len(keys))
    keys.sort()
    print(keys[:100])
    print(0 in rel_info)


if __name__ == "__main__":
    pc_rel_dev_with_cpid()
