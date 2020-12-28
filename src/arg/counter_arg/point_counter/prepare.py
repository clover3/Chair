from typing import List, Tuple

from arg.counter_arg import header
from arg.counter_arg.data_loader import load_labeled_data_per_topic
from arg.counter_arg.header import Passage


def load_data(split) -> List[Tuple[Passage, int]]:
    output_data = []
    for topic in header.topics:
        itr = load_labeled_data_per_topic(split, topic)
        for item in itr:
            p1 = item.text1, 0
            p2 = item.text2, 1
            output_data.append(p1)
            output_data.append(p2)
    return output_data



