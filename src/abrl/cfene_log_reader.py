import ast
import os
from collections import Counter
from typing import List, Tuple

import numpy as np

from cpath import data_path
from list_lib import lmap
from misc_lib import get_dir_files


def load_all_file2() -> List[Tuple[np.array, float]]:
    def parse_discrete_action(line):
        idx, action, reward = line.split("\t")
        action = np.array(ast.literal_eval(action))
        for a in action:
            if a == 0:
                raise ValueError

        reward = float(reward)
        return action, reward
    dir_path = os.path.join(data_path, "budget_allocation", "rl_logs2")

    all_items = []
    for file in get_dir_files(dir_path):
        try:
            items = lmap(parse_discrete_action, open(file, "r"))
        except ValueError:
            print(file)
            raise
        all_items.extend(items)

    counter = Counter()
    for action, reward in all_items:
        action_sanity(action, reward)
        non_neg = [v for v in action if v >= 0]
        budget100 = int(sum(non_neg) * 100 + 0.5)
        counter[budget100] += 1

    return all_items


def drop_seen(records: List[Tuple[np.array, float]]) -> List[Tuple[np.array, float]]:
    seen = set()
    output = []
    for action, reward in records:
        key = str(action)
        if key in seen:
            pass
        else:
            seen.add(key)
            output.append((action, reward))
    return output


def action_sanity(action, reward):
    assert reward < 1
    if all([a > 0.05 for a in action]):
        assert reward > 0