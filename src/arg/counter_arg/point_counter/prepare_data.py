from typing import List, Tuple

from arg.counter_arg.header import Passage
from arg.counter_arg.point_counter.prepare import load_data
from list_lib import lmap, right
from misc_lib import tprint


def get_argu_pointwise_data():
    tprint("Loading data")
    train_data: List[Tuple[Passage, int]] = load_data("training")
    dev_data = load_data("validation")

    def get_texts(e: Tuple[Passage, int]) -> str:
        return e[0].text.replace("\n", " ")

    train_x: List[str] = lmap(get_texts, train_data)
    train_y: List[int] = right(train_data)
    dev_x: List[str] = lmap(get_texts, dev_data)
    dev_y: List[int] = right(dev_data)
    return train_x, train_y, dev_x, dev_y